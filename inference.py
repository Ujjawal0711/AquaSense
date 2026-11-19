import logging
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch_ema import ExponentialMovingAverage
from torchvision import transforms
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
from torchvision.transforms import InterpolationMode

logger = logging.getLogger("EmbeddingClassifier")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "[%(asctime)s] [%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(handler)

ImageInput = Union[np.ndarray, Image.Image]


@dataclass
class PredictionResult:
    name: str
    species_id: int
    distance: float
    accuracy: float
    image_id: Optional[int]
    annotation_id: Optional[int]
    drawn_fish_id: Optional[int]


def _torch_load(path: str, map_location):
    """Load with weights_only=True (safe, default since torch 2.6); fall back for
    older local files that pickle non-tensor objects."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError:
        logger.warning(f"{path} needs full unpickling; only load files you trust.")
        return torch.load(path, map_location=map_location, weights_only=False)


class EmbeddingClassifier:
    def __init__(self, config: Dict):
        logger.setLevel(getattr(logging, config.get('log_level', 'INFO').upper()))
        logger.info("Initializing EmbeddingClassifier...")

        self._load_data(config["dataset"]["path"])
        self.dim = self.db_embeddings.shape[1]
        self._prepare_centroids()

        self.device = config["model"].get("device", "cpu")
        self._load_model(config["model"]["path"])

        self.transform = transforms.Compose([
            transforms.Resize((224, 224), InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.id_to_label = {internal_id: v['label'] for internal_id, v in self.keys.items()}
        logger.info("EmbeddingClassifier initialized successfully.")

    def _load_model(self, checkpoint_path: str):
        state_dict = _torch_load(checkpoint_path, map_location=self.device)
        # pretrained=False: the checkpoint supplies every weight.
        self.model = StableEmbeddingModel(embedding_dim=256, num_classes=639, pretrained=False, freeze_backbone=True)
        self.model.load_state_dict(state_dict['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Inference never trains, so bake the EMA weights in once. Swapping them in
        # and out per request cloned every parameter each call and was not
        # thread-safe under Flask's threaded server.
        ema = ExponentialMovingAverage(self.model.parameters(), decay=0.995)
        ema.load_state_dict(state_dict['ema_state_dict'])
        ema.copy_to(self.model.parameters())

        logger.info(f"Torch model loaded from {checkpoint_path}")
        return self.model

    def _load_data(self, dataset_path: str):
        data = _torch_load(dataset_path, map_location="cpu")
        emb = data['embeddings']
        emb = emb.numpy() if isinstance(emb, torch.Tensor) else np.asarray(emb)
        self.db_embeddings = np.ascontiguousarray(emb, dtype=np.float32)
        self.db_labels = np.array(data['labels'])
        self.image_ids = data['image_id']
        self.annotation_ids = data['annotation_id']
        self.drawn_fish_ids = data['drawn_fish_id']
        self.keys = data['labels_keys']
        self.label_to_species_id = {v['label']: v['species_id'] for v in self.keys.values()}
        logger.info(f"Dataset loaded from {dataset_path}")

    def __call__(self, img: Union[ImageInput, List[ImageInput]]):
        if isinstance(img, (np.ndarray, Image.Image)):
            return self.inference_numpy(img)
        if isinstance(img, list) and all(isinstance(i, (np.ndarray, Image.Image)) for i in img):
            return self.inference_numpy_batch(img)
        raise TypeError("Input must be np.ndarray, PIL.Image, or a list of them.")

    def _to_tensor(self, img: ImageInput) -> torch.Tensor:
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        return self.transform(img.convert("RGB"))

    def inference_numpy(self, img: ImageInput):
        return self._inference_batch_tensor(self._to_tensor(img).unsqueeze(0).to(self.device))[0]

    def inference_numpy_batch(self, imgs: List[ImageInput]):
        return self._inference_batch_tensor(torch.stack([self._to_tensor(i) for i in imgs]).to(self.device))

    def _inference_batch_tensor(self, tensors: torch.Tensor):
        with torch.inference_mode():
            embeddings, archead_logits, fc_logits = self.model(tensors)

        archead_indices = torch.argmax(archead_logits, dim=1).tolist()
        fc_indices = torch.argmax(fc_logits, dim=1).tolist()

        output = self.get_top_neighbors_from_embeddings(embeddings)
        logger.debug(f"Inference output: {output}")

        for item, arc_idx, fc_idx in zip(output, archead_indices, fc_indices):
            arc_label = self.id_to_label[arc_idx]
            fc_label = self.id_to_label[fc_idx]
            if arc_label not in item:
                item[arc_label] = {'index': None, 'similarity': 0.05, 'times': 1}
            if fc_label not in item:
                item[fc_label] = {'index': None, 'similarity': 0.01, 'times': 1}

        return self._postprocess(output)

    def _postprocess(self, class_results) -> List[List[PredictionResult]]:
        results = []
        for single_fish in class_results:
            fish_results = []
            for label, data in single_fish.items():
                index = data["index"]
                has_idx = index is not None
                fish_results.append(PredictionResult(
                    name=label,
                    species_id=self.label_to_species_id[label],
                    distance=float(data['similarity']),
                    accuracy=float(data['similarity'] / data['times']),
                    image_id=self.image_ids[index] if has_idx else None,
                    annotation_id=self.annotation_ids[index] if has_idx else None,
                    drawn_fish_id=self.drawn_fish_ids[index] if has_idx else None,
                ))
            results.append(fish_results)
        return results

    def _prepare_centroids(self):
        # One pass over the database instead of one boolean mask per class.
        unique_labels, inverse = np.unique(self.db_labels, return_inverse=True)
        sums = np.zeros((len(unique_labels), self.dim), dtype=np.float64)
        np.add.at(sums, inverse, self.db_embeddings)
        centroids = sums / np.bincount(inverse)[:, None]
        centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10

        self.centroid_matrix = centroids.astype(np.float32)
        self.centroid_labels = list(unique_labels)
        self.label_to_centroid = dict(zip(self.centroid_labels, self.centroid_matrix))

        # Precomputed member indices per class (sorted, matching database order),
        # so a query gathers candidates without scanning every label.
        order = np.argsort(inverse, kind="stable")
        bounds = np.cumsum(np.bincount(inverse))[:-1]
        self.label_to_indices = dict(zip(self.centroid_labels, np.split(order, bounds)))

    def get_top_neighbors_from_embeddings(
        self,
        query_embeddings: Union[np.ndarray, torch.Tensor],
        topk_centroid: int = 5,
        topk_neighbors: int = 10,
        centroid_threshold: float = 0.7,
        neighbor_threshold: float = 0.8
    ) -> List[Dict[str, Dict[str, Union[float, int, None]]]]:
        start_time = time.time()
        if isinstance(query_embeddings, torch.Tensor):
            query_embeddings = query_embeddings.detach().cpu().numpy()
        queries = np.ascontiguousarray(query_embeddings, dtype=np.float32)
        queries = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-10)

        # Cosine similarity to every centroid for the whole batch in one matmul.
        all_centroid_sims = queries @ self.centroid_matrix.T

        results = []
        for query_emb, centroid_sims in zip(queries, all_centroid_sims):
            k = min(topk_centroid, len(centroid_sims))
            top_idx = np.argpartition(-centroid_sims, k - 1)[:k]
            top_idx = top_idx[np.argsort(-centroid_sims[top_idx])]
            centroid_scores = {
                self.centroid_labels[i]: float(centroid_sims[i])
                for i in top_idx if centroid_sims[i] >= centroid_threshold
            }
            if not centroid_scores:
                results.append({})
                continue

            # Exact inner-product search over members of the shortlisted classes.
            # Sorted indices keep tie-breaking identical to a scan in database order.
            cand = np.sort(np.concatenate([self.label_to_indices[lbl] for lbl in centroid_scores]))
            sims = self.db_embeddings[cand] @ query_emb
            n = min(topk_neighbors, len(sims))
            top = np.argpartition(-sims, n - 1)[:n]
            top = top[np.lexsort((top, -sims[top]))]

            score_map = defaultdict(lambda: {'index': None, 'similarity': 0.0, 'times': 0})
            for j in top:
                sim = float(sims[j])
                if sim < neighbor_threshold:
                    break  # sorted descending: nothing further qualifies
                original_idx = int(cand[j])
                entry = score_map[self.db_labels[original_idx]]
                entry['similarity'] += sim
                entry['times'] += 1
                if entry['index'] is None:
                    entry['index'] = original_idx

            for label in centroid_scores:
                if label not in score_map:
                    score_map[label] = {'index': None, 'similarity': 0.1, 'times': 1}

            results.append(dict(score_map))

        logger.debug(f"Searched {len(queries)} embeddings in {time.time() - start_time:.3f}s")
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
        # `pretrained=` is deprecated in torchvision; use the weights enum.
        self.backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)

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
        emb_norm = F.normalize(self.embeddings(pooled), p=2, dim=1)
        arc_logits = self.arcface(emb_norm, labels)

        if not self.training:
            arc_logits = F.softmax(arc_logits, dim=1)
            fc_logits = F.softmax(fc_logits, dim=1)

        return emb_norm, arc_logits, fc_logits
