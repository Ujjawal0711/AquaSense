import logging
import time
from collections import defaultdict, Counter
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import faiss
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny

from PIL import Image
from scipy.stats import entropy
from sklearn.metrics import pairwise_distances
from torch_ema import ExponentialMovingAverage
from torchvision import transforms

# Setup Logger
logger = logging.getLogger("EmbeddingClassifier")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

@dataclass
class PredictionResult:
    name: str
    species_id: int
    distance: float
    accuracy: float
    image_id: int
    annotation_id: int
    drawn_fish_id: int

@contextmanager
def use_ema_params(model, ema):
    original_params = [param.detach().clone() for param in model.parameters()]
    try:
        ema.copy_to(model.parameters())
        yield
    finally:
        for param, orig in zip(model.parameters(), original_params):
            param.data.copy_(orig)

class EmbeddingClassifier:
    def __init__(self, config: Dict):
        logger.setLevel(getattr(logging, config.get('log_level', 'INFO').upper()))
        self._load_data(config["dataset"]["path"])

        self.dim = self.db_embeddings.shape[1]
        self._prepare_centroids()

        logger.info("Initializing EmbeddingClassifier...")

        self.device = config["model"].get("device", "cpu")
        self._load_model(config["model"]["path"])

        self.transform = transforms.Compose([
            transforms.Resize((224, 224), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Create ID to label mapping
        self.id_to_label = {internal_id: self.keys[internal_id]['label'] for internal_id in self.keys}

        logger.info("EmbeddingClassifier initialized successfully.")

    def _load_model(self, checkpoint_path: str):
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model = StableEmbeddingModel(embedding_dim=256, num_classes=639, pretrained=True, freeze_backbone=True)
        self.model.load_state_dict(state_dict['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.ema = ExponentialMovingAverage(self.model.parameters(), decay=0.995)
        self.ema.load_state_dict(state_dict['ema_state_dict'])

        logger.info(f"Torch model loaded from {checkpoint_path}")
        return self.model

    def _load_data(self, dataset_path: str):
        data = torch.load(dataset_path)
        self.db_embeddings = data['embeddings'].numpy().astype("float32")
        self.db_labels = np.array(data['labels'])
        self.image_ids = data['image_id']
        self.annotation_ids = data['annotation_id']
        self.drawn_fish_ids = data['drawn_fish_id']
        self.keys = data['labels_keys']
        self.label_to_species_id = {
            v['label']: v['species_id'] for v in self.keys.values()
        }
        logger.info(f"Dataset loaded from {dataset_path}")

    def __call__(self, img: Union[np.ndarray, List[np.ndarray]]):
        if isinstance(img, np.ndarray):
            return self.inference_numpy(img)
        elif isinstance(img, list) and all(isinstance(i, np.ndarray) for i in img):
            return self.inference_numpy_batch(img)
        else:
            raise TypeError("Input must be np.ndarray or List[np.ndarray].")

    def inference_numpy(self, img: np.ndarray):
        tensor = self.transform(Image.fromarray(img)).unsqueeze(0).to(self.device)
        return self._inference_batch_tensor(tensor)[0]

    def inference_numpy_batch(self, imgs: List[np.ndarray]):
        tensors = torch.stack([self.transform(Image.fromarray(img)) for img in imgs]).to(self.device)
        return self._inference_batch_tensor(tensors)

    def _inference_batch_tensor(self, tensors: torch.Tensor):
        with use_ema_params(self.model, self.ema):
            with torch.no_grad():
                embeddings, archead_logits, fc_logits = self.model(tensors)

        archead_indices = torch.argmax(archead_logits, dim=1)
        fc_indices = torch.argmax(fc_logits, dim=1)

        output = self.get_top_neighbors_from_embeddings(embeddings)
        logger.debug(f"Inference output: {output}")

        for i, (item, arc_idx, fc_idx) in enumerate(zip(output, archead_indices, fc_indices)):
            arc_label = self.id_to_label[arc_idx.item()]
            fc_label = self.id_to_label[fc_idx.item()]

            if arc_label not in item:
                item[arc_label] = {'index': None, 'similarity': 0.05, 'times': 1}

            if fc_label not in item:
                item[fc_label] = {'index': None, 'similarity': 0.01, 'times': 1}

        return self._postprocess(output)

    def _postprocess(self, class_results) -> List[PredictionResult]:
        results = []
        for single_fish in class_results:
            fish_results = []
            for label, data in single_fish.items():
                index = data["index"]
                fish_results.append(PredictionResult(
                    name=label,
                    species_id=self.label_to_species_id[label],
                    distance=data['similarity'],
                    accuracy=data['similarity'] / data['times'],
                    image_id=self.image_ids[index] if index is not None else None,
                    annotation_id=self.annotation_ids[index] if index is not None else None,
                    drawn_fish_id=self.drawn_fish_ids[index] if index is not None else None,
                ))
            results.append(fish_results)
        return results

    def _prepare_centroids(self):
        unique_labels = np.unique(self.db_labels)
        self.label_to_centroid = {}
        for label in unique_labels:
            class_embs = self.db_embeddings[self.db_labels == label]
            centroid = np.mean(class_embs, axis=0)
            centroid /= (np.linalg.norm(centroid) + 1e-10)
            self.label_to_centroid[label] = centroid

        self.centroid_matrix = np.stack([self.label_to_centroid[label] for label in self.label_to_centroid])
        self.centroid_labels = list(self.label_to_centroid.keys())

    def get_top_neighbors_from_embeddings(
        self,
        query_embeddings: Union[np.ndarray, torch.Tensor],
        topk_centroid: int = 5,
        topk_neighbors: int = 10,
        centroid_threshold: float = 0.7,
        neighbor_threshold: float = 0.8
    ) -> List[Dict[str, Dict[str, Union[float, int, None]]]]:
        start_time = time.time()
        logger.info(f"Starting search over {len(query_embeddings)} embeddings")

        if isinstance(query_embeddings, torch.Tensor):
            query_embeddings = query_embeddings.cpu().numpy().astype("float32")

        results = []
        for query_emb in query_embeddings:
            centroid_sims = 1.0 - pairwise_distances(query_emb.reshape(1, -1), self.centroid_matrix, metric='cosine')[0]
            top_centroid_indices = np.argsort(-centroid_sims)[:topk_centroid]

            centroid_scores = {
                self.centroid_labels[idx]: centroid_sims[idx]
                for idx in top_centroid_indices if centroid_sims[idx] >= centroid_threshold
            }
            selected_classes = set(centroid_scores.keys())

            if not selected_classes:
                results.append({})
                continue

            class_mask = np.isin(self.db_labels, list(selected_classes))
            selected_embeddings = self.db_embeddings[class_mask]
            selected_labels = self.db_labels[class_mask]
            selected_indices = np.where(class_mask)[0]

            if len(selected_embeddings) == 0:
                results.append({"top_neighbors": [], "centroid_scores": centroid_scores})
                continue

            faiss_index = faiss.IndexFlatIP(self.dim)
            faiss_index.add(selected_embeddings)
            distances, indices = faiss_index.search(query_emb.reshape(1, -1), min(topk_neighbors, len(selected_embeddings)))

            score_map = defaultdict(lambda: {'index': None, 'similarity': 0.0, 'times': 0})
            for rank, idx in enumerate(indices[0]):
                label = selected_labels[idx]
                sim = distances[0][rank]
                original_idx = selected_indices[idx]
                if sim >= neighbor_threshold:
                    score_map[label]['similarity'] += sim
                    score_map[label]['times'] += 1
                    if score_map[label]['index'] is None:
                        score_map[label]['index'] = original_idx

            for label, sim in centroid_scores.items():
                if label not in score_map:
                    score_map[label] = {'index': None, 'similarity': 0.1, 'times': 1}

            results.append(score_map)

        logger.info(f"Completed in {time.time() - start_time:.2f} seconds")
        return results

class AttentionPooling(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention_conv = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        weights = torch.sigmoid(self.attention_conv(x))
        weighted = x * weights
        return weighted.sum(dim=(2, 3)) / weights.sum(dim=(2, 3)).clamp(min=1e-6)


class ArcFaceHead(nn.Module):
    def __init__(self, embedding_dim, num_classes, s=30.0, m=0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m

    def forward(self, embeddings, labels=None):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        if labels is not None:
            one_hot = F.one_hot(labels, num_classes=self.weight.size(0)).float()
            theta = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))
            target_logit = torch.cos(theta + self.m)
            output = cosine * (1 - one_hot) + target_logit * one_hot
            output *= self.s
        else:
            output = cosine * self.s
        return output


class StableEmbeddingModel(nn.Module):
    def __init__(self, embedding_dim=256, num_classes=639, pretrained=True, freeze_backbone=True):
        super().__init__()
        self.backbone = convnext_tiny(pretrained=pretrained)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.pooling = AttentionPooling(in_channels=768)
        self.embeddings = nn.Linear(768, embedding_dim)

        self.fc_parallel = nn.Linear(768, num_classes)
        self.arcface = ArcFaceHead(embedding_dim, num_classes)

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x, labels=None):
        x = self.backbone.features(x)            # (B, C, H, W)
        pooled = self.pooling(x)                 # (B, 768)

        fc_logits = self.fc_parallel(pooled)
        emb = self.embeddings(pooled)
        emb_norm = F.normalize(emb, p=2, dim=1)

        if labels is not None:
            arc_logits = self.arcface(emb_norm, labels)
        else:
            arc_logits = self.arcface(emb_norm)

        if not self.training:
            arc_logits = F.softmax(arc_logits, dim=1)
            fc_logits = F.softmax(fc_logits, dim=1)

        return emb_norm, arc_logits, fc_logits

