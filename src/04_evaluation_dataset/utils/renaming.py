import os
import re
from pathlib import Path

# KONFIGURATION
# Gib hier den Pfad zu deinem Ordner mit den Originalbildern an
target_dir = "/home/team_omd/omd_fl/01_Projekte/02-EvaSteel/03_Datasets/synthetic_welds_steelbeems/test/synth_welds_good"


def rename_files(directory):
    path = Path(directory)
    if not path.exists():
        print(f"Fehler: Der Pfad {directory} existiert nicht.")
        return

    count = 0
    # Wir suchen nach allen .png Bildern
    for file in path.glob("*.png"):
        old_name = file.name

        # REGEX-ERKLÄRUNG:
        # Wir suchen nach der ersten Nummerngruppe am Anfang des Dateinamens
        # Beispiel: '0005_defect_00001_.png' -> Treffer: '0005'
        match = re.match(r"^(\d+)", old_name)

        if match:
            new_id = match.group(1)
            new_name = f"{new_id}.png"
            new_file_path = path / new_name

            # Prüfen, ob die Zieldatei schon existiert, um Überschreiben zu verhindern
            if new_file_path.exists() and old_name != new_name:
                print(f"Übersprungen: {new_name} existiert bereits.")
                continue

            try:
                file.rename(new_file_path)
                print(f"Umbenannt: {old_name}  --->  {new_name}")
                count += 1
            except Exception as e:
                print(f"Fehler beim Umbenennen von {old_name}: {e}")
        else:
            print(f"Keine Nummer am Anfang gefunden: {old_name}")

    print(f"\nFertig! {count} Dateien wurden erfolgreich angepasst.")


if __name__ == "__main__":
    rename_files(target_dir)