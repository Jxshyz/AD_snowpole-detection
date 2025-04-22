import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
from pathlib import Path
from resource_tracker import track


def data_analysis(data_path: str = "/datasets/tdt4265/ad/open/Poles/rgb", save_dir: str = "./outputs"):
    track("Data analysis")  # Start resource tracking

    RGB_PATH = Path(data_path)
    output_path = Path(save_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([p for p in RGB_PATH.glob("*.png") if p.is_file()])
    print(f"Total images found: {len(image_paths)}")

    sizes = []
    color_modes = []
    file_sizes_kb = []
    aspect_ratios = []
    corrupted_images = []

    for img_path in image_paths:
        try:
            with Image.open(img_path) as img:
                sizes.append(img.size)  # (width, height)
                color_modes.append(img.mode)
                file_sizes_kb.append(img_path.stat().st_size / 1024)
                aspect_ratios.append(img.width / img.height)
        except UnidentifiedImageError:
            corrupted_images.append(img_path)

    print(f"Corrupted images: {len(corrupted_images)}")
    if corrupted_images:
        print("Example corrupted files:", corrupted_images[:3])

    widths = [w for w, h in sizes]
    heights = [h for w, h in sizes]

    print(f"Average width: {np.mean(widths):.1f}px")
    print(f"Average height: {np.mean(heights):.1f}px")
    print(f"Width range: {min(widths)}–{max(widths)}px")
    print(f"Height range: {min(heights)}–{max(heights)}px")
    print(f"Aspect ratio range: {min(aspect_ratios):.2f}–{max(aspect_ratios):.2f}")
    print(f"Standard deviation (width): {np.std(widths):.2f}")
    print(f"Standard deviation (height): {np.std(heights):.2f}")

    unique_modes = set(color_modes)
    print(f"Color modes found: {unique_modes}")

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

    # Save 5 random sample images
    sample_paths = random.sample(image_paths, 5)
    for i, path in enumerate(sample_paths):
        try:
            with Image.open(path) as img:
                img.save(output_path / f"sample_{i+1}.png")
        except Exception as e:
            print(f"Failed to save image {path.name}: {e}")

    print(f"Saved 5 sample images to: {output_path}")
