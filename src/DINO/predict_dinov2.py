import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 1. SETUP
IMG_SIZE = 448  # Muss durch 14 teilbar sein
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "dinov2_rust_finetuned_reg.pth"


# 2. MODELL LADEN
def get_model():
    # Wir laden das Small-Modell (vits14)
    # model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
    # Gewichte laden
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


# 3. TRANSFORMS
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


def predict(image_path):
    model = get_model()
    img_orig = Image.open(image_path).convert('RGB')
    img_tensor = transform(img_orig).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        # Der offizielle Weg für DINOv2: Zwischenschichten extrahieren
        # n=1: Letzter Block
        # reshape=True: Gibt uns [Batch, Channels, Height, Width] des Grids
        # Für ViT-S/14 bei 448px ist das Grid 32x32 (448/14)
        features = model.get_intermediate_layers(img_tensor, n=4, reshape=True)[0]

        # Wir berechnen die Norm über die Feature-Dimension
        # Das zeigt uns die Regionen mit der höchsten Aktivierung (dein trainierter Rost)
        heatmap = torch.norm(features[0], dim=0).cpu().numpy()

    # Normalisierung der Heatmap für die Darstellung (0 bis 1)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    # Plotten
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))

    # Originalbild (auf Zielgröße für besseren Vergleich)
    ax[0].imshow(img_orig.resize((IMG_SIZE, IMG_SIZE)))
    ax[0].set_title("Stahlträger Original")
    ax[0].axis("off")

    # Heatmap (Magma-Farbkarte ist super für Kontraste)
    # Wir interpolieren die 32x32 Map auf 448x448 hoch
    im = ax[1].imshow(heatmap, cmap='magma', interpolation='bilinear')
    ax[1].set_title("Detektierter Rost (DINOv2 Feature-Aktivierung)")
    ax[1].axis("off")

    plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Dein Pfad
    img_path = "/home/team_omd/omd_fl/01_Projekte/02-EvaSteel/03_Datasets/01_eigener_Datensatz/Stahltraeger/rgb_images/0112.png"
    predict(img_path)