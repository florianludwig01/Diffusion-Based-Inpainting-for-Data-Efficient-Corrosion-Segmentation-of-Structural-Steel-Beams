from ultralytics import YOLO
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[4]
    dataset_root = root / "03_Datasets" / "03_synthetische_Datensatz" / "01_welds"

    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing: {data_yaml}")

    model = YOLO("yolo26s.pt")

    # Train
    model.train(
        data=str(data_yaml),
        workers=0,
        task="detect",
        imgsz=512,
        epochs=50,
        batch=2,
        patience=50,
        weight_decay=1e-3,
        project=str(dataset_root / "YOLO"),
        name="26n",
        exist_ok=True,
        degrees=10.0,  # Bild um ±10 Grad rotieren
        flipud=0.5,  # 50% Chance auf vertikales Spiegeln
        fliplr=0.5,  # 50% Chance auf horizontales Spiegeln
        mosaic=1.0,  # Mosaic-Augmentation (4 Bilder kombiniert) zu 100%
        mixup=0.2,  # 20% Chance, zwei Bilder übereinander zu blenden
        scale=0.5,  # Zoom/Skalierung ±50%
        hsv_h=0.015,  # Farbton-Variation (Hue)
        hsv_s=0.7,  # Sättigungs-Variation (Saturation)
        hsv_v=0.4
    )


    best = dataset_root / "runs" / "weights" / "best.pt"
    if best.exists():
        YOLO(str(best)).val(data=str(data_yaml), imgsz=512)

if __name__ == "__main__":
    main()
