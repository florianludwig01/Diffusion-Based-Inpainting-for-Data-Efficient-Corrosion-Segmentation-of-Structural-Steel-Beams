import cv2
import numpy as np
from pathlib import Path

# --- HIER DIE PFADE ANPASSEN ---
source_image_dir = Path(
    "/home/team_omd/omd_fl/01_Projekte/02-EvaSteel/03_Datasets/synthetic_welds_steelbeems/train/real")
target_mask_dir = Path(
    "/home/team_omd/omd_fl/01_Projekte/02-EvaSteel/03_Datasets/synthetic_welds_steelbeems/GT_masks/train/real")


def create_masks_for_folder():
    # Zielordner erstellen, falls er nicht existiert
    target_mask_dir.mkdir(parents=True, exist_ok=True)

    print(f"Verarbeite Bilder aus: {source_image_dir}")
    print(f"Masken werden gespeichert in: {target_mask_dir}")

    count = 0
    # Wir suchen alle .png Bilder im Quellordner
    for img_path in source_image_dir.glob("*.png"):
        mask_path = target_mask_dir / img_path.name

        # Falls die Maske schon da ist, überspringen wir sie
        if not mask_path.exists():
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  [Fehler] Konnte Bild nicht laden: {img_path.name}")
                continue

            # Schwarzes Bild in gleicher Größe erzeugen
            h, w = img.shape[:2]
            black_mask = np.zeros((h, w), dtype=np.uint8)

            cv2.imwrite(str(mask_path), black_mask)
            count += 1
            if count % 10 == 0:
                print(f"  {count} Masken erstellt...")

    print(f"\nFertig! Insgesamt {count} neue Masken erzeugt.")


if __name__ == "__main__":
    create_masks_for_folder()