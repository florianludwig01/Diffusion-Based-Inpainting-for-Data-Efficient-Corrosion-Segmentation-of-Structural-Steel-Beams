import os
import glob
from pathlib import Path


def merge_yolo_boxes(input_folder, output_folder, threshold=0.05):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    label_files = glob.glob(os.path.join(str(input_folder), "*.txt"))
    print(f"Gefunden: {len(label_files)} Dateien in {input_folder}")

    count_saved = 0
    for file_path in label_files:
        fname = os.path.basename(file_path)
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # WICHTIG: Leere Dateien müssen für YOLO als Background-Bilder erhalten bleiben
        if not lines or len([l for l in lines if l.strip()]) == 0:
            with open(os.path.join(output_folder, fname), 'w') as f:
                pass
            count_saved += 1
            continue

        boxes = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5: continue  # Zu kurz

            cls = int(float(parts[0]))
            coords = list(map(float, parts[1:]))

            if len(parts) == 5:  # Klassisches Box Format
                xc, yc, w, h = coords
            else:  # Polygon Format (x1, y1, x2, y2, ...)
                # Berechne Bounding Box aus Polygon-Punkten
                xs = coords[0::2]
                ys = coords[1::2]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                w, h = x_max - x_min, y_max - y_min
                xc, yc = x_min + w / 2, y_min + h / 2

            boxes.append({'cls': cls, 'x1': xc - w / 2, 'y1': yc - h / 2, 'x2': xc + w / 2, 'y2': yc + h / 2})

        # Merging Logik
        merged = []
        while boxes:
            curr = boxes.pop(0)
            has_merged = True
            while has_merged:
                has_merged = False
                remnants = []
                for other in boxes:
                    dx = max(0, max(curr['x1'], other['x1']) - min(curr['x2'], other['x2']))
                    dy = max(0, max(curr['y1'], other['y1']) - min(curr['y2'], other['y2']))
                    if dx < threshold and dy < threshold:
                        curr['x1'], curr['y1'] = min(curr['x1'], other['x1']), min(curr['y1'], other['y1'])
                        curr['x2'], curr['y2'] = max(curr['x2'], other['x2']), max(curr['y2'], other['y2'])
                        has_merged = True
                    else:
                        remnants.append(other)
                boxes = remnants
            merged.append(curr)

        with open(os.path.join(output_folder, fname), 'w') as f:
            for b in merged:
                w, h = b['x2'] - b['x1'], b['y2'] - b['y1']
                xc, yc = b['x1'] + w / 2, b['y1'] + h / 2
                f.write(f"{b['cls']} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
        count_saved += 1

    print(f"Erfolgreich gespeichert: {count_saved} Dateien in {output_folder}")

if __name__ == "__main__":
    # Pfad-Logik basierend auf deiner Struktur
    root = Path(__file__).resolve().parents[4]  # Passt sich an deine src-Tiefe an
    dataset_root = root / "03_Datasets" / "Combined_Rust"

    # Verarbeitung für Train und Val
    for split in ["train", "val"]:
        print(f"Starte Merging für {split}...")
        in_p = dataset_root / split / "labels_detailed"
        out_p = dataset_root / split / "labels"

        merge_yolo_boxes(
            input_folder=in_p,
            output_folder=out_p,
            threshold=0.12  # 5% der Bildgröße als Puffer
        )

    print("\nAlle Label-Splits wurden erfolgreich verarbeitet.")