import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


# =========================
# 1. KONFIGURATION
# =========================

IMG_SIZE = 518
BATCH_SIZE = 4
ACCUM_STEPS = 4          # effektive Batch Size = 4 * 4 = 16
EPOCHS = 20
LR = 1e-5
WEIGHT_DECAY = 0.05
NUM_WORKERS = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = "/home/team_omd/omd_fl/01_Projekte/02-EvaSteel/03_Datasets/02_Literaturdatensaetze/Corrosion/Transformer"
SAVE_PATH = "dinov2_rust_finetuned_last2blocks.pth"


# =========================
# 2. MODELL LADEN
# =========================

def load_dinov2_model():
    model = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vits14_reg"
    )
    return model


def freeze_backbone_train_last_blocks(model, num_blocks=2):
    for p in model.parameters():
        p.requires_grad = False

    for block in model.blocks[-num_blocks:]:
        for p in block.parameters():
            p.requires_grad = True

    return model


# =========================
# 3. AUGMENTIERUNG
# =========================

class DinoV2Augmentation:
    def __init__(self, size):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(
                size,
                scale=(0.7, 1.0),
                antialias=True
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
                hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
        ])

    def __call__(self, image):
        return [
            self.transform(image),
            self.transform(image)
        ]


# =========================
# 4. TRAINING
# =========================

def train():
    print(f"Nutze Gerät: {DEVICE}")

    model = load_dinov2_model().to(DEVICE)
    model = freeze_backbone_train_last_blocks(model, num_blocks=2)

    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Trainierbare Parameter: {trainable_params:,}")
    print(f"Gesamte Parameter:     {total_params:,}")

    dataset = datasets.ImageFolder(
        DATA_PATH,
        transform=DinoV2Augmentation(IMG_SIZE)
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    scaler = torch.cuda.amp.GradScaler(
        enabled=(DEVICE.type == "cuda")
    )

    print("Starte DINOv2 Fine-Tuning...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for step, (images, _) in enumerate(pbar):
            view1 = images[0].to(DEVICE, non_blocking=True)
            view2 = images[1].to(DEVICE, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                out1 = model(view1)
                out2 = model(view2)

                out1 = F.normalize(out1, dim=-1)
                out2 = F.normalize(out2, dim=-1)

                loss = 1.0 - (out1 * out2.detach()).sum(dim=-1).mean()
                loss = loss / ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * ACCUM_STEPS

            avg_loss = total_loss / (step + 1)
            pbar.set_postfix(loss=f"{avg_loss:.4f}")

        scheduler.step()

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {total_loss / len(loader):.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
        )

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Training beendet. Modell gespeichert unter: {SAVE_PATH}")


# =========================
# 5. START
# =========================

if __name__ == "__main__":
    train()