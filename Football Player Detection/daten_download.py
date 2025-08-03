import kagglehub
import shutil
import os

# Dataset herunterladen
path = kagglehub.dataset_download("iasadpanwhar/football-player-detection-yolov8")

# Zielverzeichnis
destination =r"G:\Machine_Learning\Football Player Detection"

# Sicherstellen, dass das Zielverzeichnis existiert
os.makedirs(destination, exist_ok=True)

# Alle heruntergeladenen Dateien in das Ziel kopieren
for root, dirs, files in os.walk(path):
    for file in files:
        src_file = os.path.join(root, file)
        rel_path = os.path.relpath(src_file, path)
        dest_file = os.path.join(destination, rel_path)
        os.makedirs(os.path.dirname(dest_file), exist_ok=True)
        shutil.copy2(src_file, dest_file)

print("Dateien erfolgreich kopiert nach:", destination)
