# Diffusion-Based Inpainting for Data-Efficient Corrosion Segmentation of Structural Steel Beams

Official implementation of the paper **"Diffusion-Based Inpainting for Data-Efficient Corrosion Segmentation of Structural Steel Beams"**.

**Authors:** Hossein Omid Beiki, Florian Ludwig, Patrick Dohmen, Steffen Wurm, Christian Brecher

**Affiliation:** Laboratory for Machine Tools and Production Engineering (WZL), RWTH Aachen University, Germany

**Publication:** 60th CIRP Conference on Manufacturing Systems (CIRP CMS 2027), Procedia CIRP

**Paper:** Coming soon.

## Overview

This repository provides the code and workflows accompanying our research on data-efficient corrosion segmentation of structural steel beams.

The study investigates whether diffusion-based inpainting can generate useful synthetic training images for pixel-wise corrosion segmentation when application-specific annotated data are limited.

Two diffusion-based inpainting workflows are used to generate synthetic corrosion images:

- **FLUX.1 Fill Dev**
- **Juggernaut XL v9 (SDXL)**

The generated images are evaluated using two segmentation architectures:

- **YOLO26s-seg:** An end-to-end fine-tuned segmentation model.
- **DINOv3-CNN:** A frozen pretrained DINOv3 ViT-B/16 encoder combined with a lightweight CNN decoder.

The experiments compare real-only, synthetic-only, and mixed training configurations. All models are evaluated on an independent dataset of real corroded structural steel beams.

The repository includes:

- ComfyUI workflows for synthetic corrosion generation using FLUX and SDXL.
- Training and evaluation code for YOLO26s-seg and DINOv3-CNN.
- Support for different training datasets and configurations.
- Multi-seed evaluation and statistical analysis.
- Threshold-dependent segmentation evaluation.
- Visualization and export of evaluation results.

## Installation

The training and evaluation environment is managed using [Pixi](https://pixi.sh/).

Clone the repository:

```bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_NAME>
```

Install the environment:

```bash
pixi install
```

Run all commands from the project root.

The experiments reported in the paper were performed using an NVIDIA GeForce RTX 4060 GPU with 8 GB VRAM.

### Command-Line Interface

Display all available options and their default values:

```bash
pixi run python3 src/main.py --help
```

## Datasets

The study uses real and synthetic corrosion datasets to investigate segmentation performance under different training conditions.

### Real Corrosion Datasets

**R1500**

A heterogeneous real corrosion dataset containing:

- 1,521 training images
- 141 validation images

The dataset includes corrosion images from different metallic structures, including bridges, pipelines, storage tanks, industrial machinery, and structural steel components.

**R150**

A reduced subset of the real corrosion dataset containing:

- 117 training images
- 30 validation images

This subset is used to investigate segmentation performance under limited real-data availability.

### Synthetic Corrosion Datasets

Two synthetic datasets were generated using diffusion-based inpainting.

**Synth-FLUX**

Synthetic corrosion images generated using FLUX.1 Fill Dev.

**Synth-SDXL**

Synthetic corrosion images generated using the SDXL-based Juggernaut XL v9 checkpoint.

Both datasets contain synthetic corrosion introduced into images of structural steel beams without visible corrosion.

The generated images were manually reviewed and annotated before being used for segmentation training.

### Test Dataset

**RTest**

An independent test dataset consisting of 50 RGB images acquired from three real corroded structural steel beams.

The images were captured using a Zivid 2+ M60 3D vision system under different viewing angles, illumination conditions, and backgrounds.

Corrosion regions were annotated using Label Studio with SAM-assisted pre-annotation followed by manual correction.

RTest was excluded from training.

### Data Availability

The datasets are not included in the training and evaluation commands automatically. Users must provide the corresponding dataset directories.

Dataset directory paths can be specified using:

```bash
--dataset_dir <PATH_TO_TRAINING_DATASET>
```

For evaluation:

```bash
--test_dir <PATH_TO_TEST_DATASET>
```

The dataset directory structure must follow the format expected by the corresponding data loaders.

## Synthetic Corrosion Generation

Synthetic corrosion images were generated using diffusion-based inpainting workflows implemented in [ComfyUI](https://github.com/Comfy-Org/ComfyUI).

Two different pretrained diffusion models were investigated:

1. FLUX.1 Fill Dev
2. Juggernaut XL v9 (SDXL)

The objective was to introduce realistic corrosion patterns into corrosion-free steel-beam images while preserving the original beam geometry, lighting, and surrounding scene.

The workflows differ in their masking strategies, prompting, and sampling configurations.

Consequently, the experiments evaluate the usefulness of the resulting synthetic datasets rather than isolating the influence of the underlying diffusion model.

### FLUX.1 Fill Dev

The FLUX workflow uses the pretrained FLUX.1 Fill Dev model.

For image generation, entire beam surfaces or larger surface regions were manually masked.

The positive prompt uses natural-language descriptions of corrosion patterns, including rust patches, oxidation stains, and paint degradation.

The workflow aims to preserve the geometry of the steel beam and the surrounding scene.

No negative prompt was used.

### Juggernaut XL v9 (SDXL)

The SDXL workflow uses the Juggernaut XL v9 checkpoint.

Compared with the FLUX workflow, smaller regions were manually masked to control where corrosion should appear.

The positive prompt uses descriptive keywords related to rust and weathered steel surfaces.

A negative prompt is used to suppress unwanted visual features, including illustrations, blur, text, and watermarks.

### ComfyUI Workflows

The exported ComfyUI workflows used in the experiments are included in this repository.

### Running the Workflows

1. Install ComfyUI.
2. Download the required pretrained model checkpoints.
3. Install any additional custom nodes required by the workflows.
4. Open ComfyUI.
5. Load the corresponding workflow JSON file.
6. Configure the input image and inpainting mask.
7. Adjust model paths if necessary.
8. Execute the workflow.

The workflows contain the generation settings used for the corresponding diffusion models.


### Pretrained Diffusion Models

The workflows use the following pretrained models:

**FLUX.1 Fill Dev**

https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev

**Juggernaut XL v9**

https://huggingface.co/RunDiffusion/Juggernaut-XL-v9

Please refer to the respective model providers for checkpoint availability, licensing, and usage conditions.

### Annotation of Synthetic Images

Four corrosion variants were generated for each input image by incrementing the random seed between generation runs.

After generation, the synthetic images were manually reviewed.

Corrosion segmentation masks were created using Label Studio with SAM-assisted annotation and subsequent manual correction.

The resulting annotated images form the Synth-FLUX and Synth-SDXL datasets.

## Segmentation Models

Two segmentation architectures were investigated.

### DINOv3-CNN

DINOv3-CNN combines a pretrained DINOv3 ViT-B/16 encoder with a lightweight convolutional decoder.

The encoder is frozen during training.

Spatial patch features from the final encoder layer are reshaped into a two-dimensional feature map and passed to the CNN decoder.

The decoder consists of convolutional layers with batch normalization and ReLU activation.

A final convolution produces corrosion logits, which are upsampled to the input resolution.

Binary corrosion masks are obtained using sigmoid activation and thresholding.

The decoder is trained using an equally weighted combination of binary cross-entropy and Dice losses.

### YOLO26s-seg

YOLO26s-seg is trained using the Ultralytics framework.

The model is initialized with pretrained weights and fine-tuned for corrosion instance segmentation.

For pixel-wise evaluation, predicted corrosion instance masks are combined into a single binary corrosion mask.

The default checkpoint is:

```text
yolo26s-seg.pt
```

A different checkpoint can be selected using:

```bash
--yolo_model <CHECKPOINT>
```

## Training

The following commands demonstrate how to train the segmentation models.

Replace the dataset paths with the corresponding local directories.

The dataset identifiers shown below are examples from the code interface. Refer to the experimental configurations section for the datasets used in the paper.

### Train DINOv3-CNN

```bash
pixi run python3 src/main.py \
    --mode train \
    --model dino \
    --dino_model dinov3 \
    --dataset_dir <PATH_TO_DATASET> \
    --dataset SFx1 \
    --epochs 40 \
    --batch_size 2 \
    --img_size 518 \
    --freeze_encoder \
    --early_stopping_patience 10 \
    --early_stopping_min_delta 0.0001 \
    --lr_scheduler_patience 5 \
    --lr_scheduler_factor 0.5 \
    --min_lr 0.000001
```

The `--freeze_encoder` option freezes the pretrained DINOv3 encoder.

Remove this option to fine-tune the encoder as well.

The principal training parameters reported in the paper are:

| Parameter | Value |
|---|---|
| Encoder | DINOv3 ViT-B/16 |
| Input resolution | 518 × 518 |
| Initial learning rate | 0.0001 |
| Batch size | 2 |
| Maximum epochs | 40 |
| Encoder | Frozen |
| Optimizer | AdamW |

### Train YOLO26s-seg

```bash
pixi run python3 src/main.py \
    --mode train \
    --model yolo \
    --dataset_dir <PATH_TO_DATASET> \
    --dataset SFx1 \
    --epochs 20 \
    --batch_size 2 \
    --num_workers 0 \
    --img_size 512
```

The principal training parameters reported in the paper are:

| Parameter | Value |
|---|---|
| Model | YOLO26s-seg |
| Input resolution | 512 × 512 |
| Initial learning rate | 0.0001 |
| Batch size | 2 |
| Maximum epochs | 20 |

The initial learning rate and other training settings must be configured consistently with the experiments described in the paper.

Additional training parameters can be inspected using:

```bash
pixi run python3 src/main.py --help
```

## Evaluation

Both segmentation models are evaluated on the independent real corrosion test dataset.

The evaluation supports individual checkpoints and aggregated evaluation across multiple training seeds.

### Test One DINOv3-CNN Model

```bash
pixi run python3 src/main.py \
    --mode test \
    --model dino \
    --dino_model dinov3 \
    --dataset R1 \
    --test_dir <PATH_TO_TEST_DATASET> \
    --model_path ./output/models/dinov3/R1/seed_42/best.pth \
    --output_dir ./output/test/dinov3/R1/seed_42 \
    --save_predictions
```

### Test One YOLO26s-seg Model

```bash
pixi run python3 src/main.py \
    --mode test \
    --model yolo \
    --dataset R1 \
    --test_dir <PATH_TO_TEST_DATASET> \
    --model_path ./output/models/yolo/R1/seed_42/best.pt \
    --output_dir ./output/test/yolo/R1/seed_42 \
    --img_size 512 \
    --save_predictions
```

The `--save_predictions` option saves the generated segmentation predictions.

### Evaluate Multiple Seeds

The repository supports aggregated evaluation of multiple independently trained models.

Model checkpoints must follow the directory structure:

```text
output/models/<model>/<dataset>/
├── seed_42/
│   └── best.pth
├── seed_123/
│   └── best.pth
└── ...
```

YOLO models use `best.pt` instead of `best.pth`.

#### DINOv3-CNN

```bash
pixi run python3 src/main.py \
    --mode test \
    --model dino \
    --dino_model dinov3 \
    --dataset R1 \
    --test_dir <PATH_TO_TEST_DATASET> \
    --models_dir ./output/models/dinov3/R1 \
    --all_seeds \
    --output_dir ./output/test/dinov3/R1
```

#### YOLO26s-seg

```bash
pixi run python3 src/main.py \
    --mode test \
    --model yolo \
    --dataset R1 \
    --test_dir <PATH_TO_TEST_DATASET> \
    --models_dir ./output/models/yolo/R1 \
    --all_seeds \
    --img_size 512 \
    --output_dir ./output/test/yolo/R1
```

### Evaluation Outputs

The multi-seed evaluation generates individual results for each seed and aggregated statistics.

The output files include:

```text
all_seed_results.csv
seed_statistics.csv
metric_threshold_curves.png
metric_seed_distributions.png
```

These files provide individual evaluation results, aggregated performance statistics, and visualizations of threshold-dependent performance and variation across training runs.

### Evaluation Metrics

Segmentation performance is evaluated using:

- Intersection over Union (IoU)
- Dice coefficient
- Precision
- Recall

Metrics are calculated from true-positive, false-positive, and false-negative pixel counts aggregated over the complete test dataset.

### Threshold Selection

The evaluation supports testing multiple predefined segmentation thresholds.

In the paper, 18 predefined thresholds between 0.001 and 0.9 were evaluated.

For each training run, the threshold achieving the highest IoU on RTest was selected. Dice was reported at the same threshold.

Therefore, the reported best-threshold scores represent test-set-optimized performance rather than performance at a threshold selected independently of the test data.

The threshold list can be overridden using:

```bash
--thresholds 0.1 0.2 0.3 0.4 0.5
```

Append this option to any test command.

For reproducing the paper results, use the exact threshold grid defined in the evaluation implementation.

## Experimental Configurations

The paper investigates six training configurations.

| Experiment | Training configuration | Description |
|---|---|---|
| E1 | R150 | Limited real training data |
| E2 | R1500 | Larger heterogeneous real training dataset |
| E3 | Synth-FLUX | Synthetic-only training using FLUX-generated images |
| E4 | Synth-SDXL | Synthetic-only training using SDXL-generated images |
| E5 | R1500 + Synth-FLUX | Mixed real and FLUX-generated training data |
| E6 | R1500 + Synth-SDXL | Mixed real and SDXL-generated training data |

Each configuration was evaluated using both DINOv3-CNN and YOLO26s-seg.

Ten independent training runs were performed for each model–configuration combination.

This resulted in a total of 120 training runs.

All models were evaluated on the same independent real-image test dataset, RTest.

Results are reported as the mean and sample standard deviation across the ten runs.

## Results

The following table summarizes the mean segmentation performance reported in the paper.

Scores were calculated using the IoU-optimal threshold selected separately for each training run on RTest.

| Experiment | DINOv3-CNN Dice | DINOv3-CNN IoU | YOLO26s-seg Dice | YOLO26s-seg IoU |
|---|---:|---:|---:|---:|
| E1: R150 | 0.696 | 0.534 | 0.608 | 0.439 |
| E2: R1500 | 0.708 | 0.548 | 0.611 | 0.440 |
| E3: Synth-FLUX | 0.670 | 0.504 | 0.593 | 0.424 |
| E4: Synth-SDXL | 0.387 | 0.241 | 0.374 | 0.240 |
| E5: R1500 + Synth-FLUX | 0.708 | 0.548 | **0.646** | **0.477** |
| E6: R1500 + Synth-SDXL | **0.729** | **0.574** | 0.626 | 0.456 |

### Key Findings

DINOv3-CNN achieved higher mean Dice and IoU scores than YOLO26s-seg across all evaluated training configurations, although the difference was negligible for synthetic-only SDXL training.

Increasing the amount of heterogeneous real training data resulted in only limited improvements on the target-domain test dataset.

Synthetic-only training did not consistently match the performance of real-data training.

The highest mean segmentation scores were obtained using mixed real and synthetic training data:

- **DINOv3-CNN:** R1500 + Synth-SDXL (E6)
- **YOLO26s-seg:** R1500 + Synth-FLUX (E5)

These findings indicate that the usefulness of synthetic corrosion images depends on the inpainting workflow, the segmentation architecture, and whether synthetic images replace or supplement real training data.

The mixed configurations also contained more training images than their real-only baseline. Therefore, the effects of synthetic image content and increased dataset size cannot be isolated from these experiments.

## Reproducibility

The repository provides training and evaluation code for reproducing the segmentation experiments described in the paper.

The experimental setup includes:

- Two segmentation architectures.
- Six training configurations.
- Ten independent training runs per model–configuration combination.
- A common independent real-image test dataset.
- Threshold-dependent evaluation.
- Aggregated evaluation across multiple random seeds.

The experiments were conducted using an NVIDIA GeForce RTX 4060 GPU with 8 GB VRAM.

To reproduce the reported results, ensure that the following match the experimental setup:

1. Dataset composition and train/validation/test splits.
2. Pretrained model checkpoints.
3. Training hyperparameters.
4. Random seeds.
5. Data augmentation settings.
6. Evaluation threshold grid.
7. Evaluation metric definitions.

The ComfyUI workflows provide the generation configurations used for the synthetic corrosion datasets.

Manual masking, image review, and annotation were part of the data-generation process. The synthetic-data pipeline is therefore not fully automated.

## TensorBoard

TensorBoard can be used to inspect DINOv3-CNN training logs.

Start TensorBoard using:

```bash
pixi run tensorboard --logdir ./output/tensorboard --port 6006
```

Open:

http://localhost:6006

## Citation

If you use this repository or the associated methodology in your research, please cite our paper:

**Diffusion-Based Inpainting for Data-Efficient Corrosion Segmentation of Structural Steel Beams**

Hossein Omid Beiki, Florian Ludwig, Patrick Dohmen, Steffen Wurm, Christian Brecher.

60th CIRP Conference on Manufacturing Systems (CIRP CMS 2027), Procedia CIRP.

## License

The original source code and ComfyUI workflows in this
repository are licensed under the MIT License, unless
otherwise indicated.

See the [LICENSE](LICENSE) file for details.

Third-party models, pretrained weights, datasets, and
software dependencies are subject to their respective
licenses and usage conditions.

## Acknowledgements

This work was carried out within the EvaSteel research project, funded by FOSTA – Forschungsvereinigung Stahlanwendung e.V. (project no. p 1844/13/2025 S 0024/10298/25 FE 2).
