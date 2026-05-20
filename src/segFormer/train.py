import os
import cv2
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from transformers import SegformerForSemanticSegmentation


# =========================
# KONFIGURATION
# =========================

DATASET_DIR = "/home/team_omd/omd_fl/01_Projekte/02-EvaSteel/03_Datasets/02_Literaturdatensaetze/Corrosion/Pretraining/train"

IMAGE_DIR = os.path.join(DATASET_DIR, "images")
LABEL_DIR = os.path.join(DATASET_DIR, "labels")

SAVE_PATH = "segformer_rust_best.pth"

MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
NUM_CLASSES = 2

IMG_SIZE = 512
BATCH_SIZE = 4
EPOCHS = 40
LR = 6e-5
WEIGHT_DECAY = 0.01
NUM_WORKERS = 4
VAL_SPLIT = 0.2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# YOLO-POLYGON ZU MASKE
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

        # Nur Rostklasse verwenden
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

class RustSegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, img_size=512, augment=False):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.augment = augment

        self.image_files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
        ]

        self.image_files.sort()

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

        # Resize
        image = cv2.resize(
            image,
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_LINEAR
        )

        mask = cv2.resize(
            mask,
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_NEAREST
        )

        # Augmentierungen
        if self.augment:
            if random.random() < 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()

            if random.random() < 0.3:
                alpha = random.uniform(0.85, 1.15)
                beta = random.uniform(-15, 15)
                image = np.clip(alpha * image + beta, 0, 255).astype(np.uint8)

        # Normalisierung wie ImageNet
        image = image.astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()

        return {
            "pixel_values": image,
            "labels": mask
        }


# =========================
# METRIKEN
# =========================

def compute_iou(preds, labels, num_classes=2):
    ious = []

    for cls in range(num_classes):
        pred_cls = preds == cls
        label_cls = labels == cls

        intersection = (pred_cls & label_cls).sum().item()
        union = (pred_cls | label_cls).sum().item()

        if union == 0:
            continue

        ious.append(intersection / union)

    if len(ious) == 0:
        return 0.0

    return sum(ious) / len(ious)


def compute_rust_iou(preds, labels):
    pred_rust = preds == 1
    label_rust = labels == 1

    intersection = (pred_rust & label_rust).sum().item()
    union = (pred_rust | label_rust).sum().item()

    if union == 0:
        return 0.0

    return intersection / union


# =========================
# TRAINING
# =========================

def train():
    print(f"Device: {DEVICE}")

    full_dataset = RustSegmentationDataset(
        IMAGE_DIR,
        LABEL_DIR,
        img_size=IMG_SIZE,
        augment=True
    )

    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Augmentierung für Validation deaktivieren
    val_dataset.dataset.augment = False

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

    id2label = {
        0: "background",
        1: "rust"
    }

    label2id = {
        "background": 0,
        "rust": 1
    }

    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    model.to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=(DEVICE.type == "cuda")
    )

    best_rust_iou = 0.0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for batch in pbar:
            pixel_values = batch["pixel_values"].to(DEVICE, non_blocking=True)
            labels = batch["labels"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                outputs = model(
                    pixel_values=pixel_values,
                    labels=labels
                )
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # =========================
        # VALIDATION
        # =========================

        model.eval()
        val_loss = 0.0
        mean_iou = 0.0
        rust_iou = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                pixel_values = batch["pixel_values"].to(DEVICE, non_blocking=True)
                labels = batch["labels"].to(DEVICE, non_blocking=True)

                outputs = model(
                    pixel_values=pixel_values,
                    labels=labels
                )

                loss = outputs.loss
                logits = outputs.logits

                # SegFormer gibt kleinere Logits aus, daher auf Maskengröße skalieren
                logits = F.interpolate(
                    logits,
                    size=labels.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )

                preds = torch.argmax(logits, dim=1)

                val_loss += loss.item()
                mean_iou += compute_iou(preds.cpu(), labels.cpu(), NUM_CLASSES)
                rust_iou += compute_rust_iou(preds.cpu(), labels.cpu())

        avg_val_loss = val_loss / len(val_loader)
        avg_mean_iou = mean_iou / len(val_loader)
        avg_rust_iou = rust_iou / len(val_loader)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"mIoU: {avg_mean_iou:.4f} | "
            f"Rust IoU: {avg_rust_iou:.4f}"
        )

        if avg_rust_iou > best_rust_iou:
            best_rust_iou = avg_rust_iou

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "id2label": id2label,
                    "label2id": label2id,
                    "img_size": IMG_SIZE,
                    "model_name": MODEL_NAME
                },
                SAVE_PATH
            )

            print(f"Neues bestes Modell gespeichert: Rust IoU = {best_rust_iou:.4f}")

    print("Training beendet.")
    print(f"Best Rust IoU: {best_rust_iou:.4f}")


if __name__ == "__main__":
    train()