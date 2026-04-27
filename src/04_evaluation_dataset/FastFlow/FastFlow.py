from anomalib.data import Folder
from anomalib.models import Fastflow
from anomalib.engine import Engine
from anomalib import TaskType

datamodule = Folder(
    name="weld",
    root="/home/team_omd/omd_fl/01_Projekte/02-EvaSteel/03_Datasets/synthetic_welds_steelbeems/",
    normal_dir="train/real",
    abnormal_dir="test",
    test_split_mode="from_dir",
    mask_dir="GT_masks",
)

model = Fastflow(
    backbone="resnet18",
    flow_steps=8,
)
model.pixel_metrics = None

engine = Engine(
    max_epochs=50,
    accelerator="gpu",
    devices=1,
)

engine.fit(model=model, datamodule=datamodule)
test_results = engine.test(model=model, datamodule=datamodule)
predictions = engine.predict(model=model, datamodule=datamodule)