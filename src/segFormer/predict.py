import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from transformers import SegformerForSemanticSegmentation


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT_PATH = "segformer_rust_best.pth"
IMAGE_PATH = "/home/team_omd/omd_fl/01_Projekte/02-EvaSteel/03_Datasets/01_eigener_Datensatz/Stahltraeger/rgb_images/0210.png"
OUTPUT_MASK_PATH = "rust_mask.png"
OUTPUT_OVERLAY_PATH = "rust_overlay.png"

IMG_SIZE = 512


# =========================
# MODELL LADEN
# =========================

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

model = SegformerForSemanticSegmentation.from_pretrained(
    checkpoint["model_name"],
    num_labels=2,
    id2label=checkpoint["id2label"],
    label2id=checkpoint["label2id"],
    ignore_mismatched_sizes=True
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()


# =========================
# BILD LADEN
# =========================

image_pil = Image.open(IMAGE_PATH).convert("RGB")
orig_w, orig_h = image_pil.size

image = np.array(image_pil)

image_resized = cv2.resize(
    image,
    (IMG_SIZE, IMG_SIZE),
    interpolation=cv2.INTER_LINEAR
)

image_norm = image_resized.astype(np.float32) / 255.0
image_norm = (image_norm - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])

tensor = torch.from_numpy(image_norm).permute(2, 0, 1).unsqueeze(0).float()
tensor = tensor.to(DEVICE)


# =========================
# INFERENZ
# =========================

with torch.no_grad():
    outputs = model(pixel_values=tensor)
    logits = outputs.logits

    logits = F.interpolate(
        logits,
        size=(IMG_SIZE, IMG_SIZE),
        mode="bilinear",
        align_corners=False
    )

    probs = torch.softmax(logits, dim=1)
    rust_prob = probs[0, 1].cpu().numpy()

    pred_mask = (rust_prob > 0.5).astype(np.uint8)


# =========================
# ZURÜCK AUF ORIGINALGRÖẞE
# =========================

pred_mask_orig = cv2.resize(
    pred_mask,
    (orig_w, orig_h),
    interpolation=cv2.INTER_NEAREST
)

rust_prob_orig = cv2.resize(
    rust_prob,
    (orig_w, orig_h),
    interpolation=cv2.INTER_LINEAR
)


# =========================
# SPEICHERN
# =========================

mask_vis = pred_mask_orig * 255
cv2.imwrite(OUTPUT_MASK_PATH, mask_vis)

image_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

overlay = image_bgr.copy()
overlay[pred_mask_orig == 1] = [0, 0, 255]

alpha = 0.45
result = cv2.addWeighted(image_bgr, 1 - alpha, overlay, alpha, 0)

cv2.imwrite(OUTPUT_OVERLAY_PATH, result)


# =========================
# AUSGABE
# =========================

rust_pixel_ratio = pred_mask_orig.mean() * 100

print(f"Rostfläche im Bild: {rust_pixel_ratio:.2f}%")
print(f"Maske gespeichert unter: {OUTPUT_MASK_PATH}")
print(f"Overlay gespeichert unter: {OUTPUT_OVERLAY_PATH}")