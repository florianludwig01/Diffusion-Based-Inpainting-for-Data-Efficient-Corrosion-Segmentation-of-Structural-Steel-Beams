import os
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# PFADE
# ============================================================

DATASET_DIR = Path(
    "/home/team_omd/omd_fl/01_Projekte/02-EvaSteel/03_Datasets/05_Paper/04_Synth_Jugger"
)

IMAGE_DIR = DATASET_DIR / "images"
LABEL_DIR = DATASET_DIR / "labels"
OUTPUT_DIR = DATASET_DIR / "images_with_labels"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# EINSTELLUNGEN
# ============================================================

# Transparenz der Segmentierungsmaske:
# 0.0 = unsichtbar, 1.0 = vollständig deckend
MASK_ALPHA = 0.35

# Liniendicke für Polygone und Bounding Boxes
LINE_THICKNESS = 2

# Klassenfarben im BGR-Format.
# Falls mehrere Klassen vorhanden sind, werden zusätzliche Farben
# automatisch erzeugt.
CLASS_COLORS = [
    (0, 0, 255),      # Rot
    (0, 255, 0),      # Grün
    (255, 0, 0),      # Blau
    (0, 255, 255),    # Gelb
    (255, 0, 255),    # Magenta
    (255, 255, 0),    # Cyan
]


def get_class_color(class_id: int) -> tuple[int, int, int]:
    """
    Gibt eine reproduzierbare Farbe für eine Klasse zurück.
    """
    if class_id < len(CLASS_COLORS):
        return CLASS_COLORS[class_id]

    rng = np.random.default_rng(class_id)
    color = rng.integers(50, 256, size=3)
    return tuple(int(value) for value in color)


def find_image(label_path: Path) -> Path | None:
    """
    Sucht das zum Label gehörende Bild anhand des Dateinamens.
    """
    supported_extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    ]

    for extension in supported_extensions:
        image_path = IMAGE_DIR / f"{label_path.stem}{extension}"

        if image_path.exists():
            return image_path

    return None


def draw_segmentation(
    image: np.ndarray,
    values: list[float],
    class_id: int,
) -> np.ndarray:
    """
    Zeichnet eine YOLO-Segmentierungsmaske.

    Format:
    class_id x1 y1 x2 y2 x3 y3 ...
    """
    height, width = image.shape[:2]

    coordinates = values[1:]

    if len(coordinates) < 6 or len(coordinates) % 2 != 0:
        print(
            f"Ungültiges Segmentierungslabel: "
            f"Klasse {class_id}, {len(coordinates)} Koordinaten"
        )
        return image

    points = np.array(coordinates, dtype=np.float32).reshape(-1, 2)

    # Normalisierte YOLO-Koordinaten in Pixelkoordinaten umrechnen
    points[:, 0] *= width
    points[:, 1] *= height

    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)

    points = points.astype(np.int32)

    color = get_class_color(class_id)

    overlay = image.copy()

    # Gefüllte Maske
    cv2.fillPoly(
        overlay,
        [points],
        color,
    )

    image = cv2.addWeighted(
        overlay,
        MASK_ALPHA,
        image,
        1.0 - MASK_ALPHA,
        0,
    )

    # Polygonrand
    cv2.polylines(
        image,
        [points],
        isClosed=True,
        color=color,
        thickness=LINE_THICKNESS,
        lineType=cv2.LINE_AA,
    )

    # Klassen-ID anzeigen
    x_text = int(points[:, 0].min())
    y_text = max(int(points[:, 1].min()) - 5, 20)

    cv2.putText(
        image,
        f"class {class_id}",
        (x_text, y_text),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )

    return image


def draw_bounding_box(
    image: np.ndarray,
    values: list[float],
    class_id: int,
) -> np.ndarray:
    """
    Zeichnet eine normale YOLO-Bounding-Box.

    Format:
    class_id x_center y_center width height
    """
    height, width = image.shape[:2]

    _, x_center, y_center, box_width, box_height = values

    x_center *= width
    y_center *= height
    box_width *= width
    box_height *= height

    x1 = int(x_center - box_width / 2)
    y1 = int(y_center - box_height / 2)
    x2 = int(x_center + box_width / 2)
    y2 = int(y_center + box_height / 2)

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))

    color = get_class_color(class_id)

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        LINE_THICKNESS,
    )

    cv2.putText(
        image,
        f"class {class_id}",
        (x1, max(y1 - 5, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )

    return image


def process_label_file(label_path: Path) -> bool:
    """
    Verarbeitet eine einzelne YOLO-Labeldatei.
    """
    image_path = find_image(label_path)

    if image_path is None:
        print(f"Kein Bild gefunden für: {label_path.name}")
        return False

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Bild konnte nicht gelesen werden: {image_path}")
        return False

    try:
        with open(label_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
    except OSError as error:
        print(f"Label konnte nicht gelesen werden: {label_path}")
        print(error)
        return False

    for line_number, line in enumerate(lines, start=1):
        line = line.strip()

        if not line:
            continue

        try:
            values = [float(value) for value in line.split()]
        except ValueError:
            print(
                f"Ungültige Werte in {label_path.name}, "
                f"Zeile {line_number}: {line}"
            )
            continue

        if len(values) < 5:
            print(
                f"Zu wenige Werte in {label_path.name}, "
                f"Zeile {line_number}: {line}"
            )
            continue

        class_id = int(values[0])

        # Bounding Box: genau 5 Werte
        if len(values) == 5:
            image = draw_bounding_box(
                image=image,
                values=values,
                class_id=class_id,
            )

        # Segmentierung: class_id + mindestens drei Punkte
        elif len(values) >= 7:
            image = draw_segmentation(
                image=image,
                values=values,
                class_id=class_id,
            )

        else:
            print(
                f"Unbekanntes YOLO-Format in {label_path.name}, "
                f"Zeile {line_number}: {line}"
            )

    output_path = OUTPUT_DIR / image_path.name

    success = cv2.imwrite(str(output_path), image)

    if not success:
        print(f"Ergebnis konnte nicht gespeichert werden: {output_path}")
        return False

    return True


def main() -> None:
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Der Bilderordner existiert nicht:\n{IMAGE_DIR}"
        )

    if not LABEL_DIR.exists():
        raise FileNotFoundError(
            f"Der Labelordner existiert nicht:\n{LABEL_DIR}"
        )

    label_files = sorted(LABEL_DIR.glob("*.txt"))

    if not label_files:
        print(f"Keine TXT-Labels gefunden in:\n{LABEL_DIR}")
        return

    print(f"Gefundene Labeldateien: {len(label_files)}")
    print(f"Ausgabeordner: {OUTPUT_DIR}")
    print()

    successful = 0
    failed = 0

    for index, label_path in enumerate(label_files, start=1):
        print(
            f"[{index}/{len(label_files)}] "
            f"Verarbeite {label_path.name}"
        )

        if process_label_file(label_path):
            successful += 1
        else:
            failed += 1

    print()
    print("Fertig.")
    print(f"Erfolgreich verarbeitet: {successful}")
    print(f"Fehlgeschlagen: {failed}")
    print(f"Ergebnisse gespeichert unter:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()