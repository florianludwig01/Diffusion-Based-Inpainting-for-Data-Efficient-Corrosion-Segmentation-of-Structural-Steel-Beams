import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import numpy as np


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 518
DATA_PATH = "/home/team_omd/omd_fl/01_Projekte/02-EvaSteel/03_Datasets/02_Literaturdatensaetze/Corrosion/Transformer"
SAVE_PATH = "rust_embeddings.npz"


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])


model = torch.hub.load(
    "facebookresearch/dinov2",
    "dinov2_vits14_reg"
).to(DEVICE)

model.eval()


def load_image(path):
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0)


image_paths = [
    os.path.join(DATA_PATH, f)
    for f in os.listdir(DATA_PATH)
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
]


all_embeddings = []
all_paths = []


with torch.no_grad():
    for path in tqdm(image_paths):
        x = load_image(path).to(DEVICE)

        features = model.forward_features(x)

        cls_feature = features["x_norm_clstoken"]          # Bild-Embedding
        patch_features = features["x_norm_patchtokens"]    # lokale Patch-Features

        embedding = F.normalize(cls_feature, dim=-1)

        all_embeddings.append(embedding.cpu().numpy())
        all_paths.append(path)


all_embeddings = np.concatenate(all_embeddings, axis=0)

np.savez(
    SAVE_PATH,
    embeddings=all_embeddings,
    paths=np.array(all_paths)
)

print("Gespeichert:", SAVE_PATH)
print("Embedding-Shape:", all_embeddings.shape)