from pathlib import Path

# ==========================================
# PFADE
# ==========================================

DATASET_DIR = Path(
    "/home/team_omd/omd_fl/01_Projekte/02-EvaSteel/03_Datasets/05_Paper/01_Hybrid_Flux_x1/train"
)

IMAGE_DIR = DATASET_DIR / "images"
LABEL_DIR = DATASET_DIR / "labels"

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

# ==========================================
# UMBENENNEN
# ==========================================

renamed = 0
not_found = 0

for label_path in LABEL_DIR.glob("*.txt"):

    stem = label_path.stem

    match = None

    for image_path in IMAGE_DIR.iterdir():
        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        if image_path.stem.endswith(stem):
            match = image_path
            break

    if match is None:
        print(f"❌ Kein Bild gefunden für {label_path.name}")
        not_found += 1
        continue

    new_label = LABEL_DIR / f"{match.stem}.txt"

    if new_label.exists():
        print(f"⚠ {new_label.name} existiert bereits.")
        continue

    label_path.rename(new_label)

    print(f"✓ {label_path.name} -> {new_label.name}")
    renamed += 1

print("\n===================================")
print(f"Umbenannt: {renamed}")
print(f"Nicht gefunden: {not_found}")
print("===================================")