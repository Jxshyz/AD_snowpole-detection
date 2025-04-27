from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
import csv
import time
from .resource_tracker import track

def train_yolo(
    data_yaml_path,
    model_name="yolov8n.pt",
    epochs=30,
    img_size=640,
    save_dir="./runs/simple_train"
):
    model_run = "Run #" + datetime.now().strftime("%Y%m%d-%H%M")
    print(f"🚀 Starting {model_run}...")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Start resource tracking
    stop_tracking = track("Simple Training")

    model = YOLO(model_name)

    start_time = time.time()
    model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=img_size,
        project=str(save_dir),
        name="simple_yolov8n_run",
        exist_ok=True,
        device=0,
        batch=16,
        optimizer="Adam",
        val=True,
        augment=True
    )
    end_time = time.time()
    duration = round(end_time - start_time, 2)

    # Validation metrics
    metrics = model.val()
    
    # ✅ NEW WAY: Use mean_results() from metrics.box
    precision, recall, map50, map5095 = metrics.box.mean_results()


    # ✅ Stop and log resource usage
    energy_wh, avg_memory = stop_tracking(return_stats=True)

    # ✅ Save training metrics to CSV
    log_path = Path(__file__).resolve().parent.parent / "data" / "training_summary.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_path.exists()

    with open(log_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Model Run", "Epochs", "Time (s)", "Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95", "Energy (Wh)", "Avg Memory (MB)"])
        writer.writerow([
            model_run, epochs, duration,
            f"{precision:.4f}", f"{recall:.4f}", f"{map50:.4f}", f"{map5095:.4f}",
            f"{energy_wh:.2f}", f"{avg_memory:.2f}"
        ])

    print(f"✅ Training complete. Metrics logged to {log_path}")
