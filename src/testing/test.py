import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

from models.dino import DINOv2RustSegmenter, DINOv3RustSegmenter
from data.mask_converter import yolo_seg_to_mask


def compute_metrics(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()

    eps = 1e-7

    return {
        "iou": tp / (tp + fp + fn + eps),
        "dice": (2 * tp) / (2 * tp + fp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
        "accuracy": (tp + tn) / (tp + fp + fn + tn + eps),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


class DINO_test:
    def __init__(self, args):
        self.args = args
        self.model_path = args.model_path
        self.dino_model = args.dino_model
        self.test_dir = args.test_dir
        self.image_dir = os.path.join(self.test_dir, "images")
        self.label_dir = os.path.join(self.test_dir, "labels")
        self.output_dir = args.output_dir
        self.summary_csv = os.path.join(self.output_dir, "dino_threshold_sweep.csv")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(self.output_dir, exist_ok=True)

        self.thresholds = args.thresholds

    def preprocess_image(self, image_path, img_size):
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

        return tensor, image_np, width, height

    def save_overlay(self, image_np, pred_mask, image_name, threshold):
        threshold_dir = os.path.join(
            self.output_dir,
            f"threshold_{threshold:.3f}"
        )
        os.makedirs(threshold_dir, exist_ok=True)

        base_name = os.path.splitext(image_name)[0]

        pred_mask_vis = (pred_mask * 255).astype(np.uint8)

        cv2.imwrite(
            os.path.join(threshold_dir, base_name + "_pred_mask.png"),
            pred_mask_vis
        )

        overlay = image_np.copy()
        overlay[pred_mask > 0] = [255, 0, 0]

        result = cv2.addWeighted(image_np, 0.65, overlay, 0.35, 0)
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        cv2.imwrite(
            os.path.join(threshold_dir, base_name + "_overlay.png"),
            result_bgr
        )

    def test(self):
        print("Device:", self.device)

        checkpoint = torch.load(self.model_path, map_location=self.device)

        if self.dino_model == "dinov2":
            model = DINOv2RustSegmenter(
                freeze_encoder=False
            ).to(self.device)
        elif self.dino_model == "dinov3":
            model = DINOv3RustSegmenter(freeze_encoder=False).to(self.device)
        else:
            raise NotImplementedError

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        img_size = checkpoint["img_size"]

        image_files = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
        ])

        summary_rows = []

        for threshold in self.thresholds:
            total_tp = total_fp = total_fn = total_tn = 0
            image_results = []

            for image_name in tqdm(image_files, desc=f"DINO threshold={threshold}"):
                image_path = os.path.join(self.image_dir, image_name)

                image_tensor, image_np, width, height = self.preprocess_image(
                    image_path,
                    img_size
                )

                image_tensor = image_tensor.to(self.device)

                label_name = os.path.splitext(image_name)[0] + ".txt"
                label_path = os.path.join(self.label_dir, label_name)

                gt_mask = yolo_seg_to_mask(label_path, width, height)

                with torch.no_grad():
                    logits = model(image_tensor)
                    probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

                pred_mask = (probs > threshold).astype(np.uint8)

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

                image_results.append(metrics)

                if self.args.save_predictions:
                    self.save_overlay(image_np, pred_mask, image_name, threshold)

            eps = 1e-7

            dataset_iou = total_tp / (total_tp + total_fp + total_fn + eps)
            dataset_dice = (2 * total_tp) / (2 * total_tp + total_fp + total_fn + eps)
            dataset_precision = total_tp / (total_tp + total_fp + eps)
            dataset_recall = total_tp / (total_tp + total_fn + eps)
            dataset_accuracy = (total_tp + total_tn) / (
                total_tp + total_fp + total_fn + total_tn + eps
            )

            row = {
                "threshold": threshold,
                "dataset_iou": dataset_iou,
                "dataset_dice": dataset_dice,
                "dataset_precision": dataset_precision,
                "dataset_recall": dataset_recall,
                "dataset_accuracy": dataset_accuracy,
                "mean_iou": np.mean([r["iou"] for r in image_results]),
                "mean_dice": np.mean([r["dice"] for r in image_results]),
                "mean_precision": np.mean([r["precision"] for r in image_results]),
                "mean_recall": np.mean([r["recall"] for r in image_results]),
            }

            summary_rows.append(row)

            print(
                f"Threshold {threshold} | "
                f"IoU: {dataset_iou:.4f} | "
                f"Dice: {dataset_dice:.4f} | "
                f"Precision: {dataset_precision:.4f} | "
                f"Recall: {dataset_recall:.4f}"
            )

        df = pd.DataFrame(summary_rows)
        df.to_csv(self.summary_csv, index=False)

        best = df.loc[df["dataset_iou"].idxmax()]
        print("\nBest DINO result:")
        print(best)
        print(f"\nCSV gespeichert: {self.summary_csv}")


class YOLO_test:
    def __init__(self, args):
        self.args = args
        self.model_path = args.model_path
        self.test_dir = args.test_dir
        self.image_dir = os.path.join(self.test_dir, "images")
        self.label_dir = os.path.join(self.test_dir, "labels")
        self.img_size = args.img_size
        self.output_dir = args.output_dir
        self.summary_csv = os.path.join(self.output_dir, "yolo_confidence_sweep.csv")

        os.makedirs(self.output_dir, exist_ok=True)

        self.conf_values = args.thresholds


    def predict_mask(self, model, image_path, width, height, conf):
        results = model.predict(
            source=image_path,
            imgsz=self.img_size,
            conf=conf,
            verbose=False
        )

        pred_mask = np.zeros((height, width), dtype=np.uint8)
        result = results[0]

        if result.masks is None:
            return pred_mask

        masks = result.masks.data.cpu().numpy()

        for mask in masks:
            mask_resized = cv2.resize(
                mask,
                (width, height),
                interpolation=cv2.INTER_LINEAR
            )
            pred_mask[mask_resized > 0.5] = 1

        return pred_mask

    def save_overlay(self, image, pred_mask, gt_mask, image_name, conf):
        conf_dir = os.path.join(self.output_dir, f"conf_{conf:.3f}")
        os.makedirs(conf_dir, exist_ok=True)

        base_name = os.path.splitext(image_name)[0]

        cv2.imwrite(
            os.path.join(conf_dir, base_name + "_pred_mask.png"),
            (pred_mask * 255).astype(np.uint8)
        )

        cv2.imwrite(
            os.path.join(conf_dir, base_name + "_gt_mask.png"),
            (gt_mask * 255).astype(np.uint8)
        )

        overlay_pred = image.copy()
        overlay_pred[pred_mask > 0] = [0, 0, 255]
        result_pred = cv2.addWeighted(image, 0.65, overlay_pred, 0.35, 0)

        cv2.imwrite(
            os.path.join(conf_dir, base_name + "_pred_overlay.png"),
            result_pred
        )

        overlay_gt = image.copy()
        overlay_gt[gt_mask > 0] = [0, 255, 0]
        result_gt = cv2.addWeighted(image, 0.65, overlay_gt, 0.35, 0)

        cv2.imwrite(
            os.path.join(conf_dir, base_name + "_gt_overlay.png"),
            result_gt
        )

    def test(self):
        model = YOLO(self.model_path)

        image_files = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
        ])

        summary_rows = []

        for conf in self.conf_values:
            total_tp = total_fp = total_fn = total_tn = 0
            image_results = []

            for image_name in tqdm(image_files, desc=f"YOLO conf={conf}"):
                image_path = os.path.join(self.image_dir, image_name)

                image = cv2.imread(image_path)
                if image is None:
                    continue

                height, width = image.shape[:2]

                label_name = os.path.splitext(image_name)[0] + ".txt"
                label_path = os.path.join(self.label_dir, label_name)

                gt_mask = yolo_seg_to_mask(label_path, width, height)

                pred_mask = self.predict_mask(
                    model=model,
                    image_path=image_path,
                    width=width,
                    height=height,
                    conf=conf
                )

                metrics = compute_metrics(pred_mask, gt_mask)

                total_tp += metrics["tp"]
                total_fp += metrics["fp"]
                total_fn += metrics["fn"]
                total_tn += metrics["tn"]

                image_results.append(metrics)

                if self.args.save_predictions:
                    self.save_overlay(
                        image=image,
                        pred_mask=pred_mask,
                        gt_mask=gt_mask,
                        image_name=image_name,
                        conf=conf
                    )

            eps = 1e-7

            dataset_iou = total_tp / (total_tp + total_fp + total_fn + eps)
            dataset_dice = (2 * total_tp) / (2 * total_tp + total_fp + total_fn + eps)
            dataset_precision = total_tp / (total_tp + total_fp + eps)
            dataset_recall = total_tp / (total_tp + total_fn + eps)
            dataset_accuracy = (total_tp + total_tn) / (
                total_tp + total_fp + total_fn + total_tn + eps
            )

            row = {
                "conf": conf,
                "dataset_iou": dataset_iou,
                "dataset_dice": dataset_dice,
                "dataset_precision": dataset_precision,
                "dataset_recall": dataset_recall,
                "dataset_accuracy": dataset_accuracy,
                "mean_iou": np.mean([r["iou"] for r in image_results]),
                "mean_dice": np.mean([r["dice"] for r in image_results]),
                "mean_precision": np.mean([r["precision"] for r in image_results]),
                "mean_recall": np.mean([r["recall"] for r in image_results]),
            }

            summary_rows.append(row)

            print(
                f"Conf {conf} | "
                f"IoU: {dataset_iou:.4f} | "
                f"Dice: {dataset_dice:.4f} | "
                f"Precision: {dataset_precision:.4f} | "
                f"Recall: {dataset_recall:.4f}"
            )

        df = pd.DataFrame(summary_rows)
        df.to_csv(self.summary_csv, index=False)

        best = df.loc[df["dataset_iou"].idxmax()]
        print("\nBest YOLO result:")
        print(best)
        print(f"\nCSV gespeichert: {self.summary_csv}")