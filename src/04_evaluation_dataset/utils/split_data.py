import shutil
import random
from pathlib import Path
import yaml


def split_segmentation_dataset(root_path, split_ratio=0.8):
    # Verzeichnisse definieren
    images_path = root_path / "images"
    labels_path = root_path / "labels"

    # Zielverzeichnisse erstellen
    for split in ['train', 'val']:
        (root_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (root_path / split / 'labels').mkdir(parents=True, exist_ok=True)

    # Alle Bilder finden
    image_files = list(images_path.glob('*.png'))  # oder .png
    random.shuffle(image_files)

    split_idx = int(len(image_files) * split_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    def move_files(files, target_split):
        for img_path in files:
            # Bild verschieben
            shutil.move(str(img_path), str(root_path / target_split / 'images' / img_path.name))
            # Zugehöriges Label verschieben
            label_path = labels_path / (img_path.stem + '.txt')
            if label_path.exists():
                shutil.move(str(label_path), str(root_path / target_split / 'labels' / label_path.name))

    move_files(train_files, 'train')
    move_files(val_files, 'val')
    print(f"Split fertig: {len(train_files)} Training, {len(val_files)} Validierung")
    yaml_content = {
        'path': str(dataset_root.resolve()),  # Absoluter Pfad für Stabilität
        'train': 'train/images',
        'val': 'val/images',
        'names': {0: 'weld'}  # Hier ggf. weitere Klassen ergänzen
    }

    yaml_path = dataset_root / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print(yaml_path)

root = Path(__file__).resolve().parents[4]
dataset_root = root / "03_Datasets" / "03_synthetische_Datensatz" / "01_welds"
split_segmentation_dataset(dataset_root, split_ratio=0.8)