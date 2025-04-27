from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
import csv
import time
from .resource_tracker import track

def train_yolo(data_yaml_path, model_name="yolov8n.pt", epochs=136, img_size=1080, save_dir="./runs/final_train"):
    stop_tracking = track("Final Training")  # ⏱️ Start resource tracking

    # ✅ Fixed hyperparameters
    lr = 0.001
    mosaic = 1.0

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    results_path = save_dir / "training_performance.txt"

    log_path = Path(__file__).resolve().parent.parent / "data" / "training_log.csv"
    config_path = Path(__file__).resolve().parent.parent / "data" / "training_configs.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = log_path.exists()

    run_label = "Final"
    run_name = "Final_Finetuned"

    model = YOLO(model_name)

    with open(results_path, "w") as results_file:
        results_file.write("Final Fine-tuning Training\n")
        results_file.write("======================================\n\n")

        start_time = time.time()
        model.train(
            data=data_yaml_path,
            epochs=138,              # Now 138 epochs
            imgsz=img_size,
            project=str(save_dir),
            name=run_name,
            device=0,
            batch=8,
            lr0=lr,
            mosaic=mosaic,
            fliplr=0.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            translate=0.1,
            scale=0.5,
            shear=2,
            perspective=0.001,
            optimizer="Adam",
            val=True,
            augment=True,
            save=True,
            save_period=-1,          # ✅ Save only best.pt and last.pt
            verbose=True
        )
        duration = round(time.time() - start_time, 2)

        metrics = model.val()
        precision, recall, map50, map5095 = metrics.box.mean_results()

        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M")
        csv_iter = sum(1 for _ in open(log_path)) if file_exists else 1

        with open(log_path, mode="a", newline="") as file:
            writer = csv.writer(file)
            if csv_iter == 1:
                writer.writerow(["Type", "Iteration", "Date", "Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95", "Time (s)"])
            writer.writerow([run_label, csv_iter, timestamp, f"{precision:.4f}", f"{recall:.4f}", f"{map50:.4f}", f"{map5095:.4f}", duration])

        with open(config_path, mode="a", newline="") as cfgfile:
            writer = csv.writer(cfgfile)
            if csv_iter == 1:
                writer.writerow(["Type", "Iteration", "Date", "Model", "Epochs", "Learning Rate", "Mosaic", "Optimizer"])
            writer.writerow([run_label, csv_iter, timestamp, model_name, epochs, lr, mosaic, "Adam"])

    stop_tracking()
    print(f"✅ Final fine-tuning completed. Results saved in: {results_path}")
    print(f"✅ Best and last model saved to: {save_dir / run_name / 'weights'}")
