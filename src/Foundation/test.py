import cv2
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "dinov2_rust_seg_best.pth"
IMAGE_PATH = "/home/team_omd/omd_fl/01_Projekte/02-EvaSteel/03_Datasets/01_eigener_Datensatz/Validierung Rost/images/11a5ca53-0131.png"
OUTPUT_PATH = "testbild_prediction.png"
OUTPUT_DIR = "dinov2_predictions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

THRESHOLD = 0.5


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

        features = patch_tokens.transpose(1, 2).reshape(b, -1, patch_h, patch_w)

        logits = self.decoder(features)

        logits = F.interpolate(
            logits,
            size=(h, w),
            mode="bilinear",
            align_corners=False
        )

        return logits


def preprocess_image(image_path, img_size):
    image = Image.open(image_path).convert("RGB")
    original_size = image.size

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

    return tensor, image_np, original_size


def predict():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    model_name = checkpoint["model_name"]
    img_size = checkpoint["img_size"]

    model = DINOv2RustSegmenter(model_name=model_name).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image_tensor, original_image, original_size = preprocess_image(
        IMAGE_PATH,
        img_size
    )

    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.sigmoid(logits)

    mask = probs[0, 0].cpu().numpy()
    mask = (mask > THRESHOLD).astype(np.uint8) * 255

    mask_original = cv2.resize(
        mask,
        original_size,
        interpolation=cv2.INTER_NEAREST
    )

    overlay = original_image.copy()
    overlay[mask_original > 0] = [255, 0, 0]

    result = cv2.addWeighted(original_image, 0.65, overlay, 0.35, 0)

    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(OUTPUT_PATH, result_bgr)

    print("Gespeichert:", OUTPUT_PATH)


if __name__ == "__main__":
    predict()