import sys
from utils.analysis import data_analysis
from utils.train_yolov8 import train_yolo
from utils.train_yolov8_simple import train_yolo as train_yolo_simple
from utils.submit_predictions import submit_predictions   # ⬅️ NEW IMPORT

def run_analysis():
    print("🔍 Starting data analysis for RGB snowpole images...")
    data_analysis(
        data_path="/datasets/tdt4265/ad/open/Poles/rgb",
        save_dir="./data"
    )
    print("✅ Data analysis completed. Check './data/' for results.")

def run_training():
    print("🚀 Starting YOLOv8n training with grid search...")
    train_yolo(
        data_yaml_path="./data/data.yaml",
        model_name="yolov8n.pt",
        epochs=138,
        img_size=1280,
        save_dir="./data/yolo_training"
    )
    print("✅ Training completed. Results saved in './data/yolo_training'")

def run_simple_training():
    print("⚡ Starting simplified YOLOv8n training...")
    train_yolo_simple(
        data_yaml_path="./data/data.yaml",
        model_name="yolov8n.pt",
        epochs=30,
        img_size=1280,
        save_dir="./runs/simple_train"
    )
    print("✅ Simple training completed. Results saved in './runs/simple_train'")

def run_submission():
    print("📤 Starting submission prediction and inpainting...")
    submit_predictions(
        model_path="/home/joschuas/Documents/AD_snowpole-detection/data/yolo_training/Final_Finetuned/weights/best.pt",
        source_path="/datasets/tdt4265/ad/open/Poles/rgb/images/test",
        save_project="/home/joschuas/Documents/AD_snowpole-detection/submission_files",
        save_name="final_submission"
    )
    print("✅ Submission predictions done. Check 'submission_files/final_submission'.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Please provide a mode: 'analyze', 'train', 'train_simple', or 'submit'")
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "analyze":
        run_analysis()
    elif mode == "train":
        run_training()
    elif mode == "train_simple":
        run_simple_training()
    elif mode == "submit":
        run_submission()  # ⬅️ NEW
    else:
        print(f"❌ Unknown mode: {mode}. Use 'analyze', 'train', 'train_simple', or 'submit'.")
