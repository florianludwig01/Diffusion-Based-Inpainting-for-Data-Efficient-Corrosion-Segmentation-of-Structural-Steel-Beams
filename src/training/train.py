from data.dataloader import RustDataset
from models.dino import DINOv2RustSegmenter
from loss_metrices.loss import combined_loss
from loss_metrices.metrices import rust_iou_from_logits

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import os
from ultralytics import YOLO

class DINO_train():
    def __init__(self, args):
        self.args = args
        self.dataset_dir = args.dataset_dir
        self.train_dir = os.path.join(self.dataset_dir, "train")
        self.val_dir = os.path.join(self.dataset_dir, "val")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = args.img_size
        self.dataset = args.dataset
        self.dino_model = args.dino_model
        self.output_dir = args.output_dir
        self.model_save_dir = os.path.join(self.output_dir, "models/dinov2_rust_seg/", self.dataset)
        os.makedirs(self.model_save_dir, exist_ok=True)


    def train(self, epochs, batch_size,num_workers=4, LR=1e-4, weight_decay=1e-4, freeze_encoder=True):

        train_dataset = RustDataset(
            os.path.join(self.train_dir, "images"),
            os.path.join(self.train_dir, "labels"),
            img_size=self.img_size,
            augment=True
        )

        val_dataset = RustDataset(
            os.path.join(self.val_dir, "images"),
            os.path.join(self.val_dir, "labels"),
            img_size=self.img_size,
            augment=True
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        model = DINOv2RustSegmenter(
            model_name=self.dino_model,
            freeze_encoder=freeze_encoder
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR,
            weight_decay=weight_decay
        )

        scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == "cuda"))

        best_iou = 0.0

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0

            for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epoch}"):
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
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
                    images = images.to(self.device, non_blocking=True)
                    masks = masks.to(self.device, non_blocking=True)

                    logits = model(images)
                    loss = combined_loss(logits, masks)
                    iou = rust_iou_from_logits(logits, masks)

                    val_loss += loss.item()
                    val_iou += iou.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_val_iou = val_iou / len(val_loader)

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f} | "
                f"Rust IoU: {avg_val_iou:.4f}"
            )

            if avg_val_iou > best_iou:
                best_iou = avg_val_iou

                torch.save({
                    "model_state_dict": model.state_dict(),
                    "model_name": self.dino_model,
                    "img_size": self.img_size,
                    "best_iou": best_iou
                }, os.path.join(self.model_save_dir, "best.pth"))

                print(f"Neues bestes Modell gespeichert: IoU = {best_iou:.4f}")

        print("Training beendet.")
        print("Best Rust IoU:", best_iou)

class YOLO_train:
    def __init__(self, args):
        self.args = args
        self.dataset_dir = args.dataset_dir
        self.data_yaml = os.path.join(self.dataset_dir, "data.yaml")
        self.img_size = args.img_size
        self.dataset = args.dataset
        self.output_dir = args.output_dir
        self.model_save_dir = os.path.join(self.output_dir, "models/yolo_rust_seg")
        self.yolo_model = args.yolo_model

    def train(self, epochs, batch_size, num_workers=2, LR=1e-4, weight_decay=1e-3):
        model = YOLO(self.yolo_model)

        # Train
        model.train(
            data=self.data_yaml,
            workers=num_workers,
            task="segment",
            imgsz=self.img_size,
            epochs=epochs,
            batch=batch_size,
            patience=50,
            weight_decay=weight_decay,
            project=os.path.join(self.dataset_dir, "YOLO"),
            name=str(self.dataset),
            exist_ok=True,
            degrees=10.0,  # Bild um ±10 Grad rotieren
            flipud=0.5,  # 50% Chance auf vertikales Spiegeln
            fliplr=0.5,  # 50% Chance auf horizontales Spiegeln
            mosaic=1.0,  # Mosaic-Augmentation (4 Bilder kombiniert) zu 100%
            mixup=0.2,  # 20% Chance, zwei Bilder übereinander zu blenden
            scale=0.5,  # Zoom/Skalierung ±50%
            hsv_h=0.015,  # Farbton-Variation (Hue)
            hsv_s=0.7,  # Sättigungs-Variation (Saturation)
            hsv_v=0.4
        )

        best = os.path.join(self.dataset_dir, "YOLO/weights/best.pt")
        if best.exists():
            YOLO(str(best)).val(data=str(self.data_yaml), imgsz=self.img_size)

