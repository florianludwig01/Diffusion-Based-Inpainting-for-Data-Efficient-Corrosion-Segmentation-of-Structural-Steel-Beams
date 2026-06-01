import os
import cv2
import shutil
import random
from pathlib import Path


# =========================
# PFADANPASSUNG
# =========================

SYNTH_IMAGE_DIR = Path("/home/team_omd/omd_fl/01_Projekte/02-EvaSteel/03_Datasets/03_synthetische_Datensatz/03_rust/images")
SYNTH_LABEL_DIR = Path("/home/team_omd/omd_fl/01_Projekte/02-EvaSteel/03_Datasets/03_synthetische_Datensatz/03_rust/labels")

OUT_IMAGE_DIR = Path("/home/team_omd/omd_fl/01_Projekte/02-EvaSteel/03_Datasets/03_synthetische_Datensatz/03_rust/oversampled/images")
OUT_LABEL_DIR = Path("/home/team_omd/omd_fl/01_Projekte/02-EvaSteel/03_Datasets/03_synthetische_Datensatz/03_rust/oversampled/labels")

OUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
OUT_LABEL_DIR.mkdir(parents=True, exist_ok=True)

OVERSAMPLE_FACTOR = 5

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


# =========================
# AUGMENTIERUNG
# =========================

def augment_image(image):
    aug = image.copy()

    # Helligkeit / Kontrast
    if random.random() < 0.8:
        alpha = random.uniform(0.8, 1.2)   # Kontrast
        beta = random.uniform(-25, 25)     # Helligkeit
        aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)

    # leichter Blur
    if random.random() < 0.3:
        aug = cv2.GaussianBlur(aug, (3, 3), 0)

    # leichter Noise
    if random.random() < 0.3:
        noise = np.random.normal(0, 5, aug.shape).astype("int16")
        aug = np.clip(aug.astype("int16") + noise, 0, 255).astype("uint8")

    return aug


# =========================
# MAIN
# =========================

def main():
    image_files = [
        p for p in SYNTH_IMAGE_DIR.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print("Gefundene synthetische Bilder:", len(image_files))

    for img_path in image_files:
        label_path = SYNTH_LABEL_DIR / (img_path.stem + ".txt")

        if not label_path.exists():
            print("Kein Label gefunden:", label_path)
            continue

        image = cv2.imread(str(img_path))

        if image is None:
            print("Bild konnte nicht geladen werden:", img_path)
            continue

        # Original auch kopieren
        out_img = OUT_IMAGE_DIR / f"{img_path.stem}_orig{img_path.suffix}"
        out_lbl = OUT_LABEL_DIR / f"{img_path.stem}_orig.txt"

        cv2.imwrite(str(out_img), image)
        shutil.copy(str(label_path), str(out_lbl))

        # Augmentierte Kopien erzeugen
        for i in range(OVERSAMPLE_FACTOR):
            aug = augment_image(image)

            out_img = OUT_IMAGE_DIR / f"{img_path.stem}_aug{i}{img_path.suffix}"
            out_lbl = OUT_LABEL_DIR / f"{img_path.stem}_aug{i}.txt"

            cv2.imwrite(str(out_img), aug)
            shutil.copy(str(label_path), str(out_lbl))

    print("Fertig.")
    print("Gespeichert in:")
    print(OUT_IMAGE_DIR)
    print(OUT_LABEL_DIR)


if __name__ == "__main__":
    import numpy as np
    main()