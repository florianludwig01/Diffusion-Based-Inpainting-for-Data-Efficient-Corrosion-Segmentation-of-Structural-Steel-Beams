import os
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model


# =========================
# PFADANPASSUNG
# =========================

MODEL_PATH = "dinov2_rust_seg_best.pth"

TEST_DIR = "/home/team_omd/omd_fl/01_Projekte/02-EvaSteel/03_Datasets/01_eigener_Datensatz/Validierung Rost"
IMAGE_DIR = os.path.join(TEST_DIR, "images")
LABEL_DIR = os.path.join(TEST_DIR, "labels")
OUTPUT_DIR = "dinov2_predictions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# LABEL -> MASKE
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
# MODELL
# =========================

class DINOv2RustSegmenter(nn.Module):
    def __init__(self, model_name="facebook/dinov2-base"):
        super().__init__()

        self.encoder = Dinov2Model.from_pretrained(model_name)
        hidden_dim = self.encoder.config.hidden_size

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
        patch_tokens = tokens[:, 1:, :]

        patch_h = h // 14
        patch_w = w // 14

        features = patch_tokens.transpose(1, 2).reshape(
            b, -1, patch_h, patch_w
        )

        logits = self.decoder(features)

        logits = F.interpolate(
            logits,
            size=(h, w),
            mode="bilinear",
            align_corners=False
        )

        return logits


# =========================
# PREPROCESSING
# =========================

def preprocess_image(image_path, img_size):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    image_np = np.array(image)

    image_resized = cv2.resize(
        image_np,
        (img_size, img_size),
        interpolation=cv2.INTER_LINEAR
    )

    image_resized = image_resized.astype(np.float32) / 255.0

    image_resized = (
        image_resized - np.array([0.485, 0.456, 0.406])
    ) / np.array([0.229, 0.224, 0.225])

    tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float()
    tensor = tensor.unsqueeze(0)

    return tensor, width, height


# =========================
# METRIKEN
# =========================

def compute_metrics(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()

    eps = 1e-7

    iou = tp / (tp + fp + fn + eps)
    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    accuracy = (tp + tn) / (tp + fp + fn + tn + eps)

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn
    }


# =========================
# EVALUATION
# =========================

def evaluate():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    model_name = checkpoint["model_name"]
    img_size = checkpoint["img_size"]

    model = DINOv2RustSegmenter(model_name=model_name).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_files = sorted([
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
    ])

    results = []

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    for image_name in tqdm(image_files, desc="Evaluating"):
        image_path = os.path.join(IMAGE_DIR, image_name)

        label_name = os.path.splitext(image_name)[0] + ".txt"
        label_path = os.path.join(LABEL_DIR, label_name)

        image_tensor, width, height = preprocess_image(image_path, img_size)
        image_tensor = image_tensor.to(DEVICE)

        gt_mask = yolo_seg_to_mask(label_path, width, height)

        with torch.no_grad():
            logits = model(image_tensor)
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

        pred_mask = (probs > THRESHOLD).astype(np.uint8)

        pred_mask = cv2.resize(
            pred_mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST
        )

        metrics = compute_metrics(pred_mask, gt_mask)

        total_tp += metrics["tp"]
        total_fp += metrics["fp"]
        total_fn += metrics["fn"]
        total_tn += metrics["tn"]

        results.append({
            "image": image_name,
            **metrics
        })
        # pred_mask ist 0/1
        pred_mask_vis = (pred_mask * 255).astype(np.uint8)

        base_name = os.path.splitext(image_name)[0]

        # 1. Binäre Prediction-Maske speichern
        cv2.imwrite(
            os.path.join(OUTPUT_DIR, base_name + "_pred_mask.png"),
            pred_mask_vis
        )

        # 2. Wahrscheinlichkeitskarte speichern
        prob_map = (probs * 255).astype(np.uint8)

        prob_map = cv2.resize(
            prob_map,
            (width, height),
            interpolation=cv2.INTER_LINEAR
        )

        cv2.imwrite(
            os.path.join(OUTPUT_DIR, base_name + "_prob.png"),
            prob_map
        )

        # 3. Overlay speichern
        original = cv2.imread(image_path)
        overlay = original.copy()

        # Rot markieren, OpenCV ist BGR: [0, 0, 255]
        overlay[pred_mask > 0] = [0, 0, 255]

        result = cv2.addWeighted(original, 0.65, overlay, 0.35, 0)

        cv2.imwrite(
            os.path.join(OUTPUT_DIR, base_name + "_overlay.png"),
            result
        )

    eps = 1e-7

    dataset_iou = total_tp / (total_tp + total_fp + total_fn + eps)
    dataset_dice = (2 * total_tp) / (2 * total_tp + total_fp + total_fn + eps)
    dataset_precision = total_tp / (total_tp + total_fp + eps)
    dataset_recall = total_tp / (total_tp + total_fn + eps)
    dataset_accuracy = (total_tp + total_tn) / (
        total_tp + total_fp + total_fn + total_tn + eps
    )

    print("\n===== DATASET METRIKEN =====")
    print(f"Rust IoU:      {dataset_iou:.4f}")
    print(f"Dice / F1:     {dataset_dice:.4f}")
    print(f"Precision:     {dataset_precision:.4f}")
    print(f"Recall:        {dataset_recall:.4f}")
    print(f"Pixel Accuracy:{dataset_accuracy:.4f}")

    print("\n===== MEAN IMAGE METRIKEN =====")
    print(f"Mean IoU:      {np.mean([r['iou'] for r in results]):.4f}")
    print(f"Mean Dice:     {np.mean([r['dice'] for r in results]):.4f}")
    print(f"Mean Precision:{np.mean([r['precision'] for r in results]):.4f}")
    print(f"Mean Recall:   {np.mean([r['recall'] for r in results]):.4f}")

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv("dinov2_rust_test_results.csv", index=False)

    print("\nCSV gespeichert: dinov2_rust_test_results.csv")


if __name__ == "__main__":
    evaluate()