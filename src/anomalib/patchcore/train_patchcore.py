from anomalib.data import Folder
from anomalib.models import Fastflow, Patchcore
from anomalib.engine import Engine
from anomalib import TaskType
from anomalib.metrics import AUROC, F1Score

'''
PatchCore extrahiert Merkmale (Patches) aus einem vortrainierten Backbone 
(z.B. ResNet) und speichert diese in einer "Memory Bank". Während der Vorhersage 
werden neue Bildbereiche mit diesem "Gedächtnis" verglichen. Große Abweichungen 
werden als Anomalien markiert.

"Towards Total Recall in Industrial Anomaly Detection" (Roth et al., 2022)
Link: https://arxiv.org/abs/2106.08265
'''

datamodule = Folder(
    name="beam",
    root="/home/team_omd/omd_fl/01_Projekte/02-EvaSteel/03_Datasets/02_Literaturdatensaetze/steel beams/",
    normal_dir="train_normal",
    abnormal_dir="test_abnormal",
    test_split_mode="from_dir",
    # mask_dir="GT_masks",
)

# model = Fastflow(
#     backbone="resnet18",
#     flow_steps=8,
# )
model = Patchcore(backbone="wide_resnet50_2")
model.image_metrics = [
    AUROC(fields=["pred_score", "label"]),
    F1Score(fields=["pred_score", "label"])
]

# Pixel-Metriken explizit ausschalten
model.pixel_metrics = None

engine = Engine(
    max_epochs=50,
    accelerator="gpu",
    devices=1,
    limit_val_batches=0,
    limit_predict_batches=None,
)

engine.fit(model=model, datamodule=datamodule)
#test_results = engine.test(model=model, datamodule=datamodule)
predictions = engine.predict(model=model, datamodule=datamodule)