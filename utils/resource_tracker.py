import os
import time
import csv
import datetime
import psutil
import platform
from pathlib import Path

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Constants for estimation
TDP_WATTS = 400

# ✅ Full absolute-safe path
TRACKING_PATH = Path(__file__).resolve().parent.parent / "data" / "resource_tracking.csv"
TRACKING_PATH.parent.mkdir(parents=True, exist_ok=True)

# ✅ Force-create file with headers if missing
if not TRACKING_PATH.exists():
    with open(TRACKING_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Time", "Stage", "Duration (s)", "Energy (Wh)", "Avg Memory (MB)", "GPU", "GPU Mem (MB)", "CPU"])


def _get_gpu_info():
    if not GPU_AVAILABLE:
        return "N/A", "N/A"
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return name, memory.total // (1024 * 1024)
    except Exception:
        return "Error", "Error"


def _get_cpu_info():
    return platform.processor() or platform.uname().machine


def track(stage: str):
    start_time = time.time()
    mem_before = psutil.virtual_memory().used

    print(f"[Tracking] Started: {stage}")

    def stop_tracking(return_stats=False):
        end_time = time.time()
        mem_after = psutil.virtual_memory().used

        duration = end_time - start_time
        avg_memory = ((mem_before + mem_after) / 2) / (1024 * 1024)
        energy_wh = (TDP_WATTS * duration) / 3600

        now = datetime.datetime.now()
        gpu_name, gpu_memory = _get_gpu_info()
        cpu_info = _get_cpu_info()

        row = {
            "Date": now.strftime("%Y-%m-%d"),
            "Time": now.strftime("%H:%M"),
            "Stage": stage,
            "Duration (s)": round(duration, 2),
            "Energy (Wh)": round(energy_wh, 3),
            "Avg Memory (MB)": round(avg_memory, 2),
            "GPU": gpu_name,
            "GPU Mem (MB)": gpu_memory,
            "CPU": cpu_info,
        }

        print(f"[DEBUG] Saving to: {TRACKING_PATH}")

        file_exists = TRACKING_PATH.exists()
        with open(TRACKING_PATH, mode="a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        print(f"[Tracking] Logged resource usage for: {stage}")

        if return_stats:
            return energy_wh, avg_memory

    return stop_tracking
