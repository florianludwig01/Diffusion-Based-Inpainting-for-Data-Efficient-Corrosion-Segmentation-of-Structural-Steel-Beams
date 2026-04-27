import torch
from anomalib.engine import Engine
from anomalib.data import Folder
from anomalib.models import Fastflow
from pathlib import Path


checkpoint_path = "/src/04_evaluation_dataset/FastFlow/results/Fastflow/weld/latest/weights/lightning/model.ckpt"

predict_image_dir = "/home/team_omd/omd_fl/01_Projekte/02-EvaSteel/03_Datasets/test/"

datamodule = Folder(
    name="prediction",
    root="/home/team_omd/omd_fl/01_Projekte/02-EvaSteel/03_Datasets/synthetic_welds_steelbeems/",
    normal_dir="train/real",
    normal_test_dir=predict_image_dir # Wir "täuschen" ein Testset vor
)
datamodule.setup()

model = Fastflow(backbone="resnet18")

engine = Engine(
    pixel_metrics=None,  # Wir brauchen keine Metriken für reine Vorhersage
    image_metrics=None,
    normalization="min_max", # WICHTIG: Damit die Heatmaps nicht "blau" bleiben
    visualization_settings={
        "mode": "full",      # Speichert Original, Heatmap und Overlay
        "save_images": True,
        "show_images": False
    }
)

# Vorhersage starten
print(f"Starte Vorhersage für Bilder in: {predict_image_dir}")
predictions = engine.predict(
    model=model,
    datamodule=datamodule,
    ckpt_path=checkpoint_path
)

print("\nFertig! Die Ergebnisse liegen im Ordner 'results' unter 'Fastflow'.")
