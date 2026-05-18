from ultralytics import YOLO
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[3]
    dataset_root = root / "03_Datasets" / "Corrosion" / "Finetuning"

    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing: {data_yaml}")

    model = YOLO("/home/team_omd/omd_fl/01_Projekte/02-EvaSteel/03_Datasets/Corrosion/Pretraining/YOLO/26s/weights/best.pt")

    # Train
    model.train(
        data="data.yaml",
        project=str(dataset_root / "YOLO"),
        name="26s",
        exist_ok=True,
        epochs=120,  # Etwas mehr Zeit geben, da die Lernrate niedriger sein sollte
        imgsz=512,
        batch=16,
        lr0=0.001,  # WICHTIG: 10x niedriger als Standard (0.01)
        patience=30,
        freeze=10,  # Backbone einfrieren
        mosaic=1.0,  # Bei wenig Bildern ist Mosaic essentiell
        mixup=0.1,  # Hilft gegen Overfitting
        close_mosaic=20  # Die letzten 20 Epochen ohne Mosaic für Präzision
    )


    best = dataset_root / "runs" / "weights" / "best.pt"
    if best.exists():
        YOLO(str(best)).val(data=str(data_yaml), imgsz=512)

if __name__ == "__main__":
    main()
