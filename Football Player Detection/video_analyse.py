import cv2
from ultralytics import YOLO
import os

# === Parameter anpassen ===
#model_path = r"G:\Machine_Learning\Football Player Detection\runs\vehicle-detection_20250722_093622_good_performance\model_final.pt"
video_path = r"G:\Machine_Learning\Football Player Detection\video.mp4"
output_path = r"G:\Machine_Learning\Football Player Detection\detected_video.mp4"
conf_threshold = 0.3  # Konfidenzschwelle für Erkennungen

# === Modell laden ===
print("🔍 Lade YOLO-Modell...")
model = YOLO("yolov8l.pt")  # Wenn du keine Custom-Klassen brauchst


# === Video öffnen ===
print("🎥 Lade Video...")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"❌ Konnte Video nicht öffnen: {video_path}")

# === Video-Writer vorbereiten ===
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print("🚀 Starte Inferenz auf Video...")

# === Frameweise Verarbeitung ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 🔍 Vorhersage mit YOLO
    results = model.predict(frame, conf=conf_threshold, verbose=False)

    # 📦 Bounding Boxes einzeichnen
    annotated_frame = results[0].plot()

    # 💾 In Ausgabedatei schreiben
    out.write(annotated_frame)

    # 🖼️ Optional: Live-Vorschau
    cv2.imshow("YOLOv8 Video Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Aufräumen ===
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Video mit Erkennungen gespeichert unter:\n{output_path}")
