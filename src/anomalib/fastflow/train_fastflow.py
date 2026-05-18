from anomalib.data import Folder
from anomalib.models import Fastflow, Patchcore
from anomalib.engine import Engine
from anomalib import TaskType
from anomalib.metrics import AUROC, F1Score
from pathlib import Path
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype
import torch

'''
FastFlow nutzt ein 2D-Normalizing-Flow-Netzwerk, um die Verteilung der Merkmale 
einer normalen Stahloberfläche in eine einfache Standard-Gauß-Verteilung zu 
transformieren (zu "flattern"). 
1. Beim Training lernt das Modell, wie "guter" Stahl mathematisch aussieht.
2. Bei der Vorhersage erkennt es Anomalien daran, dass sie sich nicht in diese 
   Standard-Verteilung pressen lassen.
3. Es liefert sehr glatte und hochauflösende Heatmaps.

ORIGINAL PAPER:
"FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows" 
(Yu et al., 2021)
Link: https://arxiv.org/abs/2111.07677
'''

base_path = Path(__file__).resolve()
dataset_root = "/home/team_omd/omd_fl/01_Projekte/02-EvaSteel/03_Datasets/02_Literaturdatensaetze/steel beams/"
results_dir = base_path.parent.parent.parent.parent / "results/"

backbone = "deit_base_distilled_patch16_384" # also possible: cait_m48_448, deit_base_distilled_patch16_384, resnet18, wide_resnet50_2

cait_transform = Compose([
    Resize((448, 448), antialias=True),
    ToImage(),
    ToDtype(torch.float32, scale=True),
])

datamodule = Folder(
    name="beam",
    root=dataset_root,
    normal_dir="train_normal",
    abnormal_dir="test_abnormal",
    test_split_mode="from_dir",
    augmentations=cait_transform,
    # mask_dir="GT_masks",
)

model = Fastflow(backbone=backbone)
# model.image_metrics = [
#     AUROC(fields=["pred_score", "label"]),
#     F1Score(fields=["pred_score", "label"])
# ]
#
# # Pixel-Metriken explizit ausschalten
# model.pixel_metrics = None
datamodule.image_size = (448, 448)
model.model.input_size = (448, 448)

engine = Engine(
    default_root_dir=results_dir,
    max_epochs=50,
    accelerator="gpu",
    devices=1,
    limit_val_batches=0,
    limit_predict_batches=None,
)

engine.fit(model=model, datamodule=datamodule)
#test_results = engine.test(model=model, datamodule=datamodule)
predictions = engine.predict(model=model, datamodule=datamodule)