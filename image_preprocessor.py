# This combines logic from utils.py and dataset.py
import numpy as np
import numbers
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms.functional import pad
from PIL import Image


def get_padding(image):
    w, h = image.size
    max_wh = np.max([w, h])
    h_padding = (max_wh - w) / 2
    v_padding = (max_wh - h) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding + 0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding + 0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding - 0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding - 0.5
    padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    return padding


class NewPad:
    def __init__(self, fill=0, padding_mode='constant'):
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        return pad(img, get_padding(img), self.fill, self.padding_mode)


# This is the complete transformation pipeline
def preprocess_image(image_bytes):
    # Open the image
    image = Image.open(image_bytes).convert('RGB')

    # 1. Pad to square
    padding_transform = NewPad()
    padded_image = padding_transform(image)

    # Convert back to numpy array for albumentations
    image_np = np.array(padded_image)

    # 2. Apply Resize, Normalization, and Tensor conversion
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    transformed = transform(image=image_np)
    return transformed['image']