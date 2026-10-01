from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Configuration
# -----------------------------

DATA_DIR = Path("data/raw/NEU-CLS")
RESULTS_DIR = Path("results/plots")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_MAP = {
    "Cr": "Crazing",
    "In": "Inclusion",
    "Pa": "Patches",
    "PS": "Pitted Surface",
    "RS": "Rolled-in Scale",
    "Sc": "Scratches"
}

# -----------------------------
# Count images
# -----------------------------

class_counts = {}

for prefix, class_name in CLASS_MAP.items():

    files = list(DATA_DIR.glob(f"{prefix}_*.bmp"))

    class_counts[class_name] = len(files)

print("\nDATASET SUMMARY")
print("=" * 45)

for class_name, count in class_counts.items():
    print(f"{class_name:20s}: {count}")

print("=" * 45)
print("Total images:", sum(class_counts.values()))


# -----------------------------
# Class distribution
# -----------------------------

plt.figure(figsize=(10, 6))

bars = plt.bar(
    class_counts.keys(),
    class_counts.values()
)

plt.title("NEU-CLS Class Distribution")
plt.xlabel("Defect Class")
plt.ylabel("Number of Images")

plt.xticks(rotation=25)

for bar, value in zip(bars, class_counts.values()):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 5,
        str(value),
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "class_distribution.png",
    dpi=300
)

plt.close()


# -----------------------------
# Sample images
# -----------------------------

fig, axes = plt.subplots(
    2,
    3,
    figsize=(12, 8)
)

for ax, (prefix, class_name) in zip(
    axes.flatten(),
    CLASS_MAP.items()
):

    files = list(DATA_DIR.glob(f"{prefix}_*.bmp"))

    image = Image.open(files[0])

    ax.imshow(image, cmap="gray")
    ax.set_title(class_name)
    ax.axis("off")

plt.suptitle(
    "Representative Samples from NEU-CLS Dataset",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "sample_images.png",
    dpi=300
)

plt.close()


# -----------------------------
# Image dimensions
# -----------------------------

sample_file = list(DATA_DIR.glob("*.bmp"))[0]

image = Image.open(sample_file)

print("\nIMAGE INFORMATION")
print("=" * 45)
print("Sample image:", sample_file.name)
print("Image size:", image.size)
print("Image mode:", image.mode)
print("Image format:", image.format)
print("=" * 45)


# -----------------------------
# Pixel intensity distribution
# -----------------------------

pixels = np.array(image)

plt.figure(figsize=(10, 6))

plt.hist(
    pixels.flatten(),
    bins=50
)

plt.title("Pixel Intensity Distribution")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    RESULTS_DIR / "pixel_distribution.png",
    dpi=300
)

plt.close()


print("\nEDA COMPLETED SUCCESSFULLY!")
print("Charts saved in:", RESULTS_DIR)
