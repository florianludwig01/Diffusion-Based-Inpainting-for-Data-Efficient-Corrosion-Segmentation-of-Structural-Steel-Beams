from data.dataloader import RustDataset, seed_worker
from models.dino import DINOv2RustSegmenter, DINOv3RustSegmenter
from loss_metrices.loss import combined_loss
from loss_metrices.metrices import rust_iou_from_logits

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import os
from ultralytics import YOLO
from pathlib import Path
import shutil
import random
import numpy as np



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


class DINO_train():
    def __init__(self, args):
        self.args = args
        self.dataset_dir = args.dataset_dir
        self.train_dir = os.path.join(self.dataset_dir, "train")
        self.val_dir = os.path.join(self.dataset_dir, "valid")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = args.img_size
        self.dataset = args.dataset
        self.dino_model = args.dino_model
        self.output_dir = args.output_dir
        self.seed = args.seed
        self.model_save_dir = os.path.join(
            self.output_dir, "models", self.dino_model, self.dataset, f"seed_{self.seed}"
        )
        os.makedirs(self.model_save_dir, exist_ok=True)


    def train(self, epochs, batch_size,num_workers=4, LR=1e-4, weight_decay=1e-4, freeze_encoder=True):

        set_seed(self.seed)
        generator = torch.Generator()
        generator.manual_seed(self.seed)

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
            augment=False
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=generator
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=generator
        )
        if self.dino_model == "dinov2":
            model = DINOv2RustSegmenter(
                freeze_encoder=freeze_encoder
            ).to(self.device)
        elif self.dino_model == "dinov3":
            model = DINOv3RustSegmenter(freeze_encoder=freeze_encoder).to(self.device)
        else:
            raise NotImplementedError

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
                    "best_iou": best_iou,
                    "seed": self.seed
                }, os.path.join(self.model_save_dir, "best.pth"))

                print(f"Saved new best model: IoU = {best_iou:.4f}")

        print("Training finished.")
        print("Best Rust IoU:", best_iou)

class YOLO_train:
    def __init__(self, args):
        self.args = args
        self.dataset_dir = Path(args.dataset_dir).resolve()
        self.data_yaml = self.dataset_dir / "data.yaml"
        self.img_size = args.img_size
        self.dataset = args.dataset
        self.output_dir = Path(args.output_dir).resolve()
        self.yolo_model = args.yolo_model
        self.seed = args.seed

        self.project_dir = self.output_dir / "YOLO"
        self.model_save_dir = (
            self.output_dir / "models" / "yolo_rust_seg" / self.dataset / f"seed_{self.seed}"
        )

        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)

    def train(self, epochs, batch_size, num_workers=2, LR=1e-4, weight_decay=1e-3):
        set_seed(self.seed)
        model = YOLO(self.yolo_model)

        model.train(
            data=str(self.data_yaml),
            workers=num_workers,
            seed=self.seed,
            deterministic=True,
            task="segment",
            imgsz=self.img_size,
            epochs=epochs,
            batch=batch_size,
            patience=50,
            weight_decay=weight_decay,
            project=str(self.project_dir),
            name=f"{self.dataset}_seed_{self.seed}",
            exist_ok=False,
            lr0=LR,

            degrees=10.0,
            flipud=0.5,
            fliplr=0.5,
            mosaic=0.0,
            mixup=0.1,
            scale=0.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
        )

        best_dst = self.model_save_dir / f"best.pt"
        shutil.copy2(model.trainer.best, best_dst)
