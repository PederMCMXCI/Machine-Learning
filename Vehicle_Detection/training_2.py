import os
import yaml
import torch
from ultralytics import YOLO
import datetime
import matplotlib.pyplot as plt
import cv2
import shutil

# === Bildvorschau-Generator ===
def plot_val_images(image_dir, save_path, grid_size=(3, 3), image_size=320):
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                   if f.lower().endswith((".jpg", ".png"))]
    image_paths = sorted(image_paths)[:grid_size[0] * grid_size[1]]

    fig, axs = plt.subplots(grid_size[0], grid_size[1], figsize=(grid_size[1]*3, grid_size[0]*3))

    for ax, img_path in zip(axs.ravel(), image_paths):
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(os.path.basename(img_path), fontsize=6)
        ax.axis('off')

    for ax in axs.ravel()[len(image_paths):]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# === Hauptfunktion ===
def main():
    project_root = r"G:\Machine_Learning\PVehicleDetection-YOLOv8"
    dataset_path = os.path.join(project_root, "VehiclesDetectionDataset")
    class_names = ['Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck']

    splits = {
        'train': os.path.join(dataset_path, 'train'),
        'val': os.path.join(dataset_path, 'valid')
    }

    print("🔍 Überprüfung der Bild-/Label-Dateien...")
    for split, path in splits.items():
        img_path = os.path.join(path, "images")
        lbl_path = os.path.join(path, "labels")

        if not os.path.exists(img_path) or not os.path.exists(lbl_path):
            print(f"⚠️  Fehlender Ordner: {img_path} oder {lbl_path}")
            continue

        images = sorted([f for f in os.listdir(img_path) if f.lower().endswith((".jpg", ".png"))])
        labels = sorted([f for f in os.listdir(lbl_path) if f.endswith(".txt")])
        missing_labels = [img for img in images if not os.path.exists(os.path.join(lbl_path, os.path.splitext(img)[0] + ".txt"))]

        print(f"📁 Split '{split}': {len(images)} Bilder, {len(labels)} Labels")
        if missing_labels:
            print(f"🚫 Fehlende Labels: {missing_labels}")

    yaml_data = {
        'train': os.path.join(dataset_path, 'train', 'images').replace("\\", "/"),
        'val': os.path.join(dataset_path, 'valid', 'images').replace("\\", "/"),
        'nc': len(class_names),
        'names': class_names
    }

    yaml_path = os.path.join(project_root, "data.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)

    print(f"\n✅ data.yaml erstellt unter:\n📄 {yaml_path}")

    if torch.cuda.is_available():
        device = 0
        device_name = torch.cuda.get_device_name(0)
    else:
        device = "cpu"
        device_name = "CPU"
    print(f"\n💻 Trainingsgerät: {device_name}")

    print("\n🚀 Starte YOLOv8 Training...")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"vehicle-detection_{timestamp}"

    model = YOLO("yolov8l.pt")
    model.train(
        data=yaml_path,
        epochs=50,
        imgsz=960,
        batch=16,
        device=device,
        project=os.path.join(project_root, "runs"),
        name=run_name,
        exist_ok=False,
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=3,
        warmup_bias_lr=0.1,
        cos_lr=True,
        degrees=5.0,
        translate=0.2,
        scale=0.5,
        mosaic=0.3,
        mixup=0.2,
        #multi_scale=True
    )

    print(f"\n✅ Training abgeschlossen. Ergebnisse unter:\n{os.path.join(project_root, 'runs', run_name)}")

    print("\n📊 Starte automatische Validierung...")

    val_output_dir = os.path.join(project_root, "runs", run_name, "val")
    model.val(
        data=yaml_path,
        batch=16,
        imgsz=960,
        device=device,
        save=True,
        project=val_output_dir,
        name="",
        exist_ok=True
    )

    print("\n✅ Validierung abgeschlossen. Ergebnisse unter:")
    print(val_output_dir)

    # === 6. Vorschau-Collage erzeugen ===
    print("\n🖼️ Erstelle saubere Vorschau-Bildcollage...")

    label_img_dir = os.path.join(val_output_dir, "labels")
    preview_output = os.path.join(val_output_dir, "val_preview.png")

    if os.path.exists(label_img_dir):
        plot_val_images(label_img_dir, preview_output)
        print(f"✅ Vorschau gespeichert unter:\n{preview_output}")
    else:
        print("⚠️ Kein Label-Ordner für Vorschau gefunden.")

    # === 7. Modell speichern ===
    print("\n💾 Speichere bestes Modell im Run-Ordner...")

    weights_dir = os.path.join(project_root, "runs", run_name, "weights")
    best_model_path = os.path.join(weights_dir, "best.pt")
    final_model_path = os.path.join(project_root, "runs", run_name, "model_final.pt")

    if os.path.exists(best_model_path):
        shutil.copy(best_model_path, final_model_path)
        print(f"✅ Modell gespeichert als: {final_model_path}")
    else:
        print("❌ Konnte best.pt nicht finden. Wurde das Training erfolgreich abgeschlossen?")


if __name__ == "__main__":
    main()
