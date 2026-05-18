import torch
from anomalib.engine import Engine
from anomalib.data import Folder
from anomalib.models import Fastflow, Patchcore
from pathlib import Path


checkpoint_path = "/home/team_omd/omd_fl/01_Projekte/02-EvaSteel/04_Evaluation_Dataset/src/04_evaluation_dataset/FastFlow/results/Patchcore/beam/v1/weights/lightning/model.ckpt"

predict_image_dir = "/home/team_omd/omd_fl/01_Projekte/02-EvaSteel/03_Datasets/test/"

datamodule = Folder(
    name="prediction",
    root="/home/team_omd/omd_fl/01_Projekte/02-EvaSteel/03_Datasets/test/",
    normal_dir="rgb_images",
    normal_test_dir=predict_image_dir # Wir "täuschen" ein Testset vor
)
datamodule.setup()

# model = Fastflow(backbone="resnet18")
model = Patchcore(backbone="wide_resnet50_2")

engine = Engine(
    max_epochs=50,
    accelerator="gpu",
    devices=1,
    limit_val_batches=0,
    limit_predict_batches=None,
)

# Vorhersage starten
print(f"Starte Vorhersage für Bilder in: {predict_image_dir}")
predictions = engine.predict(
    model=model,
    datamodule=datamodule,
    ckpt_path=checkpoint_path
)

print("\nFertig! Die Ergebnisse liegen im Ordner 'results' unter 'Fastflow'.")
