# Evaluation Dataset

Training and evaluation of DINOv3 and YOLO segmentation models for rust detection.

## Setup

Run all commands from the project root. Install the environment with:

```bash
pixi install
```

The commands below use project-specific absolute dataset paths. Adjust them if the datasets are stored elsewhere.

## Train DINOv3

```bash
pixi run python3 src/main.py \
    --mode train \
    --model dino \
    --dino_model dinov3 \
    --dataset_dir /home/team_omd/omd_fl/01_Projekte/02-EvaSteel/03_Datasets/05_Paper/02_Synth_FLUX_SF/01_Synth_FLUX_SFx1 \
    --dataset SFx1 \
    --epochs 30 \
    --batch_size 4 \
    --img_size 518 \
    --freeze_encoder \
    --early_stopping_patience 10 \
    --early_stopping_min_delta 0.0001 \
    --lr_scheduler_patience 5 \
    --lr_scheduler_factor 0.5 \
    --min_lr 0.000001
```

Remove `--freeze_encoder` to fine-tune the encoder as well.

## Train YOLO

```bash
pixi run python3 src/main.py \
    --mode train \
    --model yolo \
    --dataset_dir /home/team_omd/omd_fl/01_Projekte/02-EvaSteel/03_Datasets/05_Paper/02_Synth_FLUX_SF/01_Synth_FLUX_SFx1 \
    --dataset SFx1 \
    --epochs 20 \
    --batch_size 2 \
    --num_workers 0 \
    --img_size 512
```

The default YOLO checkpoint is `yolo26s-seg.pt`. Select another checkpoint with `--yolo_model`.

## Test one DINOv3 model

```bash
pixi run python3 src/main.py \
    --mode test \
    --model dino \
    --dino_model dinov3 \
    --dataset R1 \
    --test_dir /home/team_omd/omd_fl/01_Projekte/02-EvaSteel/03_Datasets/05_Paper/00_Test_Data_Real \
    --model_path ./output/models/dinov3/R1/seed_42/best.pth \
    --output_dir ./output/test/dinov3/R1/seed_42 \
    --save_predictions
```

## Test one YOLO model

```bash
pixi run python3 src/main.py \
    --mode test \
    --model yolo \
    --dataset R1 \
    --test_dir /home/team_omd/omd_fl/01_Projekte/02-EvaSteel/03_Datasets/05_Paper/00_Test_Data_Real \
    --model_path ./output/models/yolo/R1/seed_42/best.pt \
    --output_dir ./output/test/yolo/R1/seed_42 \
    --img_size 512 \
    --save_predictions
```

## Evaluate all seeds

The model directory must have the following structure:

```text
output/models/<model>/<dataset>/
├── seed_42/best.pth
├── seed_123/best.pth
└── ...
```

YOLO models use `best.pt` instead of `best.pth`.

### DINOv3

```bash
pixi run python3 src/main.py \
    --mode test \
    --model dino \
    --dino_model dinov3 \
    --dataset R1 \
    --test_dir /home/team_omd/omd_fl/01_Projekte/02-EvaSteel/03_Datasets/05_Paper/00_Test_Data_Real \
    --models_dir ./output/models/dinov3/R1 \
    --all_seeds \
    --output_dir ./output/test/dinov3/R1
```

### YOLO

```bash
pixi run python3 src/main.py \
    --mode test \
    --model yolo \
    --dataset R1 \
    --test_dir /home/team_omd/omd_fl/01_Projekte/02-EvaSteel/03_Datasets/05_Paper/00_Test_Data_Real \
    --models_dir ./output/models/yolo/R1 \
    --all_seeds \
    --img_size 512 \
    --output_dir ./output/test/yolo/R1
```

The multi-seed evaluation writes individual seed results as well as the aggregated files `all_seed_results.csv`, `seed_statistics.csv`, `metric_threshold_curves.png`, and `metric_seed_distributions.png`.

## Select thresholds

Testing uses a predefined threshold list. Override it with, for example:

```bash
--thresholds 0.1 0.2 0.3 0.4 0.5
```

Append this option to any test command.

## TensorBoard

Start TensorBoard for the DINO training logs:

```bash
pixi run tensorboard --logdir ./output/tensorboard --port 6006
```

Then open <http://localhost:6006>.

## Command-line help

Display every available option and its default value:

```bash
pixi run python3 src/main.py --help
```

When splitting a command across several lines, every continued line except the last must end with `\`. There must be no spaces after it.
