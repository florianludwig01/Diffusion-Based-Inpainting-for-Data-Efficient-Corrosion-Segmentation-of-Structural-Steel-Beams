import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.hub
from tqdm import tqdm

# 1. KONFIGURATION
IMG_SIZE = 224
BATCH_SIZE = 4  # DINOv2 ViT-S ist recht effizient, 4-8 sollten auf den meisten GPUs gehen
EPOCHS = 15  # Bei 1000 Bildern reichen oft 10-20 Epochen für den Domain-Transfer
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "/home/team_omd/omd_fl/01_Projekte/02-EvaSteel/03_Datasets/02_Literaturdatensaetze/Corrosion/Pretraining/train/images"


# 2. DINOv2 MODEL ÜBER HUB LADEN
def load_dinov2_model():
    # Wir nutzen 'dinov2_vits14' (Small Version mit 14er Patches)
    # DINOv2 ist sehr stark im Verständnis von Texturen (wie Rost)
    # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
    return model


# 3. DINOv2 SPEZIFISCHE AUGMENTIERUNG
# Wir behalten das Konzept bei, zwei Ansichten zu vergleichen (Self-Distillation)
class DinoV2Augmentation:
    def __init__(self, size):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.3, 1.0), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __call__(self, image):
        return [self.transform(image), self.transform(image)]


# 4. TRAINING FUNKTION
def train():
    model = load_dinov2_model().to(DEVICE)

    # Dataset & Loader
    # ImageFolder erwartet Unterordner (z.B. DATA_PATH/images/...)
    dataset = datasets.ImageFolder(DATA_PATH, transform=DinoV2Augmentation(IMG_SIZE))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # Optimizer
    # Für DINOv2 nutzen wir eine sehr kleine Lernrate, um die starken Gewichte nicht zu zerstören
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=0.01)

    # Cosine Annealing Scheduler (hilft Transformern beim Konvergieren)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"Starte DINOv2 Fine-tuning auf {DEVICE}...")
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for images, _ in pbar:
            view1, view2 = images[0].to(DEVICE), images[1].to(DEVICE)

            optimizer.zero_grad()

            # Forward pass für beide Sichten
            # DINOv2 gibt direkt die Features aus
            out1 = model(view1)
            out2 = model(view2)

            # Feature Alignment Loss (Cosine Similarity)
            # Das Modell lernt, dass beide Ansichten desselben Rost-Patches identisch sind
            loss = 1 - torch.nn.functional.cosine_similarity(out1, out2).mean()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss / (pbar.n + 1))

        scheduler.step()

    # Speichern der Gewichte
    torch.save(model.state_dict(), "dinov2_rust_finetuned_reg.pth")
    print("Training erfolgreich beendet und Modell gespeichert.")


if __name__ == "__main__":
    train()