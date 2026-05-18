from ultralytics import YOLO

# 1. Dein finetuned Modell laden
# model = YOLO('/home/team_omd/omd_fl/01_Projekte/02-EvaSteel/03_Datasets/01_eigener_Datensatz/Stahltraeger/YOLO_training/beam_seg/weights/best.pt')
model = YOLO('/home/team_omd/omd_fl/01_Projekte/02-EvaSteel/03_Datasets/02_Literaturdatensaetze/Corrosion/Pretraining/YOLO/26s/weights/best.pt')

# 2. Inference auf dem Video ausführen
results = model.track(
    source='/home/team_omd/omd_fl/01_Projekte/02-EvaSteel/03_Datasets/01_eigener_Datensatz/Videotest/test.mov',
    conf=0.4,
    save=True,
    stream=True,
    project='/home/team_omd/omd_fl/01_Projekte/02-EvaSteel/03_Datasets/01_eigener_Datensatz/Videotest/',
    name='Video_Analyse_Rost',
    exist_ok=True
)

for r in results:
    pass 