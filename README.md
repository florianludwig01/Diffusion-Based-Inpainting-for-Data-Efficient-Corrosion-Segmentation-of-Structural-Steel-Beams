# EvaSteel - Anomalieerkennung an Stahlträgern

Dieses Repository enthält die Implementierung zur automatisierten Defekterkennung auf Stahloberflächen mittels der Anomalib-Bibliothek. Dieser Branch dient der Evaluation und dem Vergleich der Modelle PatchCore und FastFlow.

## Projektübersicht
Ziel ist die Detektion von Oberflächenanomalien (Rissbildung, Korrosion, Verformung) an Stahlbauteilen. Das Projekt nutzt Unsupervised Learning Ansätze, bei denen das Modell lediglich mit fehlerfreien Bildern trainiert wird.

### Implementierte Modelle

1. PatchCore
- Funktionsweise: Extrahiert Merkmale über ein vortrainiertes Backbone (z.B. ResNet) und speichert diese in einer hocheffizienten Memory Bank (Coreset).
- Vorteil: Extrem hohe Genauigkeit bei sehr geringer Anzahl an Trainingsbildern.
- Referenz: "Towards Total Recall in Industrial Anomaly Detection" (Roth et al., 2022)
- Link: https://arxiv.org/abs/2106.08265

2. FastFlow
- Funktionsweise: Nutzt 2D-Normalizing-Flows, um die Verteilung der Merkmale in einen Standard-Gauß-Raum zu transformieren.
- Vorteil: Liefert sehr präzise, hochauflösende Wahrscheinlichkeitskarten der Anomalien.
- Referenz: "FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows" (Yu et al., 2021)
- Link: https://arxiv.org/abs/2111.07677

## Installation und Umgebung

Das Projekt verwendet pixi zur Paketverwaltung.

1. Umgebung aktivieren:
   pixi shell

2. Abhängigkeiten sicherstellen:
   Die benötigten Bibliotheken (torch, anomalib, pytorch-lightning) sind in der pixi.toml definiert.

## Datenstruktur
Das Folder-Datamodule erwartet die Daten lokal unter folgendem Pfad:

steel_beams/
├── train_normal/    # Bilder für das Training (nur fehlerfrei)
└── test_abnormal/   # Bilder für die Validierung/Vorhersage (mit Defekten)

## Nutzung

### Training und Evaluierung
Zum Starten des Trainingsprozesses wird die Engine-Klasse von Anomalib genutzt. Die Parameter für Backbones (z.B. resnet18 oder wide_resnet50_2) können im Skript angepasst werden.

### Vorhersage (Inferenz)
Die Vorhersage erfolgt über den engine.predict Aufruf. Hierbei wird ein Pfad zu einem Modell-Checkpoint (.ckpt) sowie das Datamodule übergeben.

## Ergebnisse und Visualisierung
Die Ergebnisse werden standardmäßig im Ordner 'results' gespeichert.

- Image: Das ursprüngliche Eingabebild.
- Anomaly Map: Heatmap der berechneten Anomalie-Scores.
- Pred Mask: Binäre Maske, die Bereiche oberhalb des berechneten Thresholds markiert.

Hinweis: Bei PatchCore hängen die Ergebnisse stark von der Wahl des Backbones ab. Bei FastFlow ist eine ausreichende Anzahl an Epochen (mind. 50-100) entscheidend, um Artefakte in der Heatmap zu vermeiden.