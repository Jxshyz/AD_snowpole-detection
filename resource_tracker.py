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
TDP_WATTS = 15
TRACKING_PATH = Path("./data/resource_tracking.csv")
TRACKING_PATH.parent.mkdir(parents=True, exist_ok=True)


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

    # Record initial memory
    mem_before = psutil.virtual_memory().used

    print(f"[Tracking] Started: {stage}")
    input("[Tracking] Press ENTER to stop tracking...")  # Manual endpoint

    end_time = time.time()
    mem_after = psutil.virtual_memory().used

    duration = end_time - start_time  # seconds
    avg_memory = ((mem_before + mem_after) / 2) / (1024 * 1024)  # in MB
    energy_wh = (TDP_WATTS * duration) / 3600  # Convert to watt-hours

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

    file_exists = TRACKING_PATH.exists()

    with open(TRACKING_PATH, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"[Tracking] Logged resource usage for: {stage}")
