import os
import cv2
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from transformers import Dinov2Model


# =========================
# KONFIGURATION
# =========================

DATASET_DIR = "/home/team_omd/omd_fl/01_Projekte/02-EvaSteel/03_Datasets/02_Literaturdatensaetze/Corrosion/Pretraining/train"

IMAGE_DIR = os.path.join(DATASET_DIR, "images")
LABEL_DIR = os.path.join(DATASET_DIR, "labels")

SAVE_PATH = "dinov2_rust_seg_best.pth"

DINO_MODEL = "facebook/dinov2-base"
IMG_SIZE = 518  # wichtig: durch Patchgröße 14 teilbar
BATCH_SIZE = 2
EPOCHS = 40
LR = 1e-4
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.2
NUM_WORKERS = 4

FREEZE_ENCODER = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# YOLO SEGMENTATION LABEL -> MASKE
# =========================

def yolo_seg_to_mask(label_path, width, height):
    mask = np.zeros((height, width), dtype=np.uint8)

    if not os.path.exists(label_path):
        return mask

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        values = line.strip().split()
        if len(values) < 7:
            continue

        cls_id = int(float(values[0]))
        if cls_id != 0:
            continue

        coords = list(map(float, values[1:]))
        points = []

        for i in range(0, len(coords), 2):
            x = int(coords[i] * width)
            y = int(coords[i + 1] * height)
            points.append([x, y])

        points = np.array(points, dtype=np.int32)

        if len(points) >= 3:
            cv2.fillPoly(mask, [points], 1)

    return mask


# =========================
# DATASET
# =========================

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


# =========================
# DINOv2 + DECODER
# =========================

class DINOv2RustSegmenter(nn.Module):
    def __init__(self, model_name="facebook/dinov2-base", freeze_encoder=True):
        super().__init__()

        self.encoder = Dinov2Model.from_pretrained(model_name)
        hidden_dim = self.encoder.config.hidden_size

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        outputs = self.encoder(pixel_values=x)
        tokens = outputs.last_hidden_state

        # CLS token entfernen
        patch_tokens = tokens[:, 1:, :]

        patch_h = h // 14
        patch_w = w // 14

        features = patch_tokens.transpose(1, 2).reshape(b, -1, patch_h, patch_w)

        logits = self.decoder(features)

        logits = F.interpolate(
            logits,
            size=(h, w),
            mode="bilinear",
            align_corners=False
        )

        return logits


# =========================
# LOSS: BCE + DICE
# =========================

def dice_loss_with_logits(logits, targets, smooth=1.0):
    probs = torch.sigmoid(logits)

    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (union + smooth)

    return 1.0 - dice.mean()


def combined_loss(logits, targets):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss_with_logits(logits, targets)
    return bce + dice


# =========================
# METRIK
# =========================

def rust_iou_from_logits(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = probs > threshold
    targets = targets > 0.5

    intersection = (preds & targets).sum().float()
    union = (preds | targets).sum().float()

    if union == 0:
        return torch.tensor(0.0)

    return intersection / union


# =========================
# TRAINING
# =========================

def train():
    print("Device:", DEVICE)

    base_dataset = RustDataset(
        IMAGE_DIR,
        LABEL_DIR,
        img_size=IMG_SIZE,
        augment=False
    )

    indices = list(range(len(base_dataset)))
    random.seed(42)
    random.shuffle(indices)

    val_size = int(len(indices) * VAL_SPLIT)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_dataset_full = RustDataset(
        IMAGE_DIR,
        LABEL_DIR,
        img_size=IMG_SIZE,
        augment=True
    )

    val_dataset_full = RustDataset(
        IMAGE_DIR,
        LABEL_DIR,
        img_size=IMG_SIZE,
        augment=False
    )

    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    model = DINOv2RustSegmenter(
        model_name=DINO_MODEL,
        freeze_encoder=FREEZE_ENCODER
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    best_iou = 0.0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images = images.to(DEVICE, non_blocking=True)
            masks = masks.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                logits = model(images)
                loss = combined_loss(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        val_iou = 0.0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validation"):
                images = images.to(DEVICE, non_blocking=True)
                masks = masks.to(DEVICE, non_blocking=True)

                logits = model(images)
                loss = combined_loss(logits, masks)
                iou = rust_iou_from_logits(logits, masks)

                val_loss += loss.item()
                val_iou += iou.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Rust IoU: {avg_val_iou:.4f}"
        )

        if avg_val_iou > best_iou:
            best_iou = avg_val_iou

            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": DINO_MODEL,
                "img_size": IMG_SIZE,
                "best_iou": best_iou
            }, SAVE_PATH)

            print(f"Neues bestes Modell gespeichert: IoU = {best_iou:.4f}")

    print("Training beendet.")
    print("Best Rust IoU:", best_iou)


if __name__ == "__main__":
    train()