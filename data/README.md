# Published data

## RTest

RTest contains 50 RGB images of three corroded steel beams and their corrosion annotations. These are the images used for evaluation in the paper.

- `RTest/images/`: original evaluation images.
- `RTest/labels/`: corrosion annotations in YOLO segmentation format.

Each annotation file has the same basename as its corresponding image. Class 0 denotes corrosion. Each line contains a class ID followed by normalized polygon coordinates.

## Synthetic examples

`synthetic_examples/` contains an example subset, not the complete synthetic training datasets.

- `RSource/`: source images used for the published examples.
- `Synth-FLUX/images/`: synthetic images generated using FLUX.1 Fill Dev.
- `Synth-FLUX/labels/`: corresponding corrosion annotations.
- `Synth-SDXL/images/`: synthetic images generated using Juggernaut XL v9.
- `Synth-SDXL/labels/`: corresponding corrosion annotations.

The complete Synth-FLUX and Synth-SDXL datasets each contain 147 images, with 117 used for training and 30 for validation.

The complete source and synthetic image collections are not redistributed because redistribution rights remain unresolved for some source images.

## Sources and licensing

File-specific attribution and licensing information is provided in `sources_and_licenses.csv`. The repository's MIT license applies to the original code and workflows, not automatically to the images and annotations.
