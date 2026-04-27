import os
from pathlib import Path

label_dir = Path("/home/team_omd/omd_fl/01_Projekte/02-EvaSteel/03_Datasets/corrosion detect/val/labels")

for label_file in label_dir.glob("*.txt"):
    with open(label_file, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.split()
        if not parts: continue

        # 1. Klasse auf 0 setzen (falls sie 1 war)
        parts[0] = "0"

        # 2. Negative Koordinaten auf 0 clippen
        coords = [str(max(0.0, min(1.0, float(x)))) for x in parts[1:]]

        new_lines.append(" ".join([parts[0]] + coords) + "\n")

    with open(label_file, 'w') as f:
        f.writelines(new_lines)

print("Labels bereinigt: Klasse auf 0 gesetzt und Koordinaten begrenzt.")