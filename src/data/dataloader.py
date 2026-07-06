import os
import cv2
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from .mask_converter import yolo_seg_to_mask

class RustDataset(Dataset):
    def __init__(self, image_dir, label_dir, img_size=518, augment=False):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.augment = augment

        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)

        label_name = os.path.splitext(image_name)[0] + ".txt"
        label_path = os.path.join(self.label_dir, label_name)

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        mask = yolo_seg_to_mask(label_path, width, height)

        image = np.array(image)

        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            if random.random() < 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()

            if random.random() < 0.4:
                alpha = random.uniform(0.8, 1.2)
                beta = random.uniform(-20, 20)
                image = np.clip(alpha * image + beta, 0, 255).astype(np.uint8)

            if random.random() < 0.3:
                image = cv2.GaussianBlur(image, (3, 3), 0)

        image = image.astype(np.float32) / 255.0

        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).float().unsqueeze(0)

        return image, mask