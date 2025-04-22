import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError, ImageDraw
from pathlib import Path
from .resource_tracker import track
from tqdm import tqdm


def data_analysis(data_path: str = "/datasets/tdt4265/ad/open/Poles/rgb", save_dir: str = "./data"):
    stop = track("Data analysis")  # Start resource tracking (non-blocking)

    RGB_PATH = Path(data_path)
    output_path = Path(save_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([p for p in (RGB_PATH / "images").rglob("*.PNG") if p.is_file()])
    print(f"Total images found: {len(image_paths)}")

    sizes = []
    color_modes = []
    file_sizes_kb = []
    aspect_ratios = []
    corrupted_images = []

    for img_path in tqdm(image_paths, desc="Analyzing image metadata"):
        try:
            with Image.open(img_path) as img:
                sizes.append(img.size)
                color_modes.append(img.mode)
                file_sizes_kb.append(img_path.stat().st_size / 1024)
                aspect_ratios.append(img.width / img.height)
        except UnidentifiedImageError:
            corrupted_images.append(img_path)

    widths = [w for w, h in sizes]
    heights = [h for w, h in sizes]

    print(f"Corrupted images: {len(corrupted_images)}")
    print(f"Average width: {np.mean(widths):.1f}px")
    print(f"Average height: {np.mean(heights):.1f}px")
    print(f"Aspect ratio range: {min(aspect_ratios):.2f}–{max(aspect_ratios):.2f}")
    print(f"Color modes found: {set(color_modes)}")

    # Save histogram plots
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.hist(file_sizes_kb, bins=30, color="skyblue")
    plt.title("File Size Distribution (KB)")
    plt.xlabel("Size (KB)")
    plt.ylabel("Count")

    plt.subplot(1, 3, 2)
    plt.hist(aspect_ratios, bins=30, color="salmon")
    plt.title("Aspect Ratio Distribution")
    plt.xlabel("W / H")
    plt.ylabel("Count")

    plt.subplot(1, 3, 3)
    plt.hist(widths, bins=30, color="limegreen")
    plt.title("Image Width Distribution")
    plt.xlabel("Pixels")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig(output_path / "histograms.png")
    print(f"Saved histogram summary to {output_path / 'histograms.png'}")

    # Label analysis and Ground Truth Box visualization
    label_counts = []
    box_widths = []
    box_heights = []

    # Only sample from valid/ and test/ where label files exist
    valid_test_dirs = ["valid", "test"]
    sample_paths = []

    for split in valid_test_dirs:
        img_dir = RGB_PATH / "images" / split
        label_dir = RGB_PATH / "labels" / split

        images = sorted([p for p in img_dir.glob("*.PNG") if p.is_file()])
        for img_path in images:
            label_path = label_dir / (img_path.stem + ".txt")
            if label_path.exists():
                sample_paths.append((img_path, label_path))

    sample_paths = random.sample(sample_paths, min(5, len(sample_paths)))

    for img_path, label_path in tqdm(sample_paths, desc="Analyzing labels & saving samples"):
        try:
            with Image.open(img_path) as img:
                draw = ImageDraw.Draw(img)
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    label_counts.append(len(lines))
                    for line in lines:
                        cls, x_center, y_center, w_rel, h_rel = map(float, line.strip().split())
                        img_w, img_h = img.size
                        w = w_rel * img_w
                        h = h_rel * img_h
                        x = (x_center * img_w) - w / 2
                        y = (y_center * img_h) - h / 2
                        draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
                        box_widths.append(w)
                        box_heights.append(h)
                img.save(output_path / f"sample_with_boxes_{img_path.name}")
        except Exception as e:
            print(f"Failed on {img_path.name}: {e}")

    # Save histogram of bounding box counts
    if label_counts:
        plt.figure()
        plt.hist(label_counts, bins=range(0, max(label_counts) + 2), color="orange", edgecolor="black", align="left")
        plt.title("Number of Bounding Boxes per Image")
        plt.xlabel("Number of Boxes")
        plt.ylabel("Image Count")
        plt.savefig(output_path / "label_distribution.png")
        print(f"Saved label distribution to {output_path / 'label_distribution.png'}")

    # Box size distribution
    if box_widths and box_heights:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.hist(box_widths, bins=30, color="blue")
        plt.title("Bounding Box Widths (px)")

        plt.subplot(1, 2, 2)
        plt.hist(box_heights, bins=30, color="green")
        plt.title("Bounding Box Heights (px)")

        plt.tight_layout()
        plt.savefig(output_path / "box_size_distribution.png")
        print(f"Saved box size distribution to {output_path / 'box_size_distribution.png'}")

    print(f"Saved {len(sample_paths)} sample images with boxes to: {output_path}")

    # Write summary file
    summary_path = output_path / "data_analysis_findings.txt"
    with open(summary_path, "w") as f:
        f.write("📊 Data Analysis Summary\n")
        f.write("=======================\n\n")
        f.write(f"Total images found: {len(image_paths)}\n")
        f.write(f"Image width: mean = {np.mean(widths):.1f}px, std = {np.std(widths):.1f}px\n")
        f.write(f"Image height: mean = {np.mean(heights):.1f}px, std = {np.std(heights):.1f}px\n")
        f.write(f"Aspect ratio: min = {min(aspect_ratios):.2f}, max = {max(aspect_ratios):.2f}\n")
        f.write(f"Color modes found: {set(color_modes)}\n")
        f.write(f"Corrupted images: {len(corrupted_images)}\n\n")

        f.write("🧠 Bounding Box Statistics (from sampled valid/test images)\n")
        f.write("-----------------------------------------------------------\n")
        f.write(f"Sampled images: {len(sample_paths)}\n")
        f.write(f"Label files found: {len(label_counts)}\n")
        f.write(f"Average number of boxes per sampled image: {np.mean(label_counts) if label_counts else 0:.2f}\n")
        f.write(f"Bounding Box width (px): mean = {np.mean(box_widths) if box_widths else 0:.1f}, std = {np.std(box_widths) if box_widths else 0:.1f}\n")
        f.write(f"Bounding Box height (px): mean = {np.mean(box_heights) if box_heights else 0:.1f}, std = {np.std(box_heights) if box_heights else 0:.1f}\n")

    print(f"📄 Saved summary to {summary_path}")

    stop()  # ✅ End tracking + log results
