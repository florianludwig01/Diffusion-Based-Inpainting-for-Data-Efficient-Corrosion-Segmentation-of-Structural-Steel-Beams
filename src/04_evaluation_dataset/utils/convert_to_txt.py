import cv2
import os
from pathlib import Path


def convert_folder_to_yolo(input_dir, output_dir, class_id=0):
    # Zielordner erstellen, falls er nicht existiert
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Alle gängigen Bildformate berücksichtigen
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(valid_extensions):
            mask_path = os.path.join(input_dir, filename)

            # Dateiname ohne Endung extrahieren (z.B. "bild1" statt "bild1.png")
            file_stem = Path(filename).stem
            output_path = os.path.join(output_dir, f"{file_stem}.txt")

            # Maske einlesen
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None: continue

            height, width = mask.shape
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            yolo_bboxes = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)

                # YOLO-Berechnung (Mittelpunkt & Größe normalisiert)
                x_center = (x + w / 2) / width
                y_center = (y + h / 2) / height
                norm_width = w / width
                norm_height = h / height

                yolo_bboxes.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")

            # Datei speichern (auch wenn sie leer ist, falls keine Kontur gefunden wurde)
            with open(output_path, "w") as f:
                f.write("\n".join(yolo_bboxes))

            print(f"Konvertiert: {filename} -> {file_stem}.txt")


# --- Konfiguration ---
INPUT_FOLDER = "/home/team_omd/omd_fl/01_Projekte/02-EvaSteel/03_Datasets/03_synthetische_Datensatz/GT_masks/synth_welds_good"
OUTPUT_FOLDER = "/home/team_omd/omd_fl/01_Projekte/02-EvaSteel/03_Datasets/03_synthetische_Datensatz/labels"
DEFAULT_CLASS = 0

convert_folder_to_yolo(INPUT_FOLDER, OUTPUT_FOLDER, class_id=DEFAULT_CLASS)