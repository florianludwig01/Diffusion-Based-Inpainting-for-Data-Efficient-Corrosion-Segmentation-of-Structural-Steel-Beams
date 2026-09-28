from pathlib import Path
import random
import shutil
import yaml

dataset_root = Path(
    "/home/team_omd/omd_fl/01_Projekte/02-EvaSteel/03_Datasets/05_Paper/04_Synth_Jugger"
)

images_dir = dataset_root / "images"
labels_dir = dataset_root / "labels"

out_dir = dataset_root / "yolo_split"

train_ratio = 0.8
seed = 42

class_names = ["corrosion"]  

for split in ["train", "val"]:
    (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
images = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in image_exts])

pairs = []
missing_labels = []

for img in images:
    label = labels_dir / f"{img.stem}.txt"
    if label.exists():
        pairs.append((img, label))
    else:
        missing_labels.append(img.name)

random.seed(seed)
random.shuffle(pairs)

n_train = int(len(pairs) * train_ratio)
train_pairs = pairs[:n_train]
val_pairs = pairs[n_train:]

def copy_pairs(pairs, split):
    for img, label in pairs:
        shutil.copy2(img, out_dir / "images" / split / img.name)
        shutil.copy2(label, out_dir / "labels" / split / label.name)

copy_pairs(train_pairs, "train")
copy_pairs(val_pairs, "val")

data_yaml = {
    "path": str(out_dir.resolve()),
    "train": "images/train",
    "val": "images/val",
    "names": {i: name for i, name in enumerate(class_names)}
}

with open(out_dir / "data.yaml", "w", encoding="utf-8") as f:
    yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)

print(f"Fertig.")
print(f"Train: {len(train_pairs)} Bilder")
print(f"Val:   {len(val_pairs)} Bilder")
print(f"data.yaml: {out_dir / 'data.yaml'}")

if missing_labels:
    print("\nWarnung: Bilder ohne Label:")
    for name in missing_labels:
        print(" -", name)
