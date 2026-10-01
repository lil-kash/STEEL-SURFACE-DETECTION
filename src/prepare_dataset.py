from pathlib import Path
import shutil
import random

# -----------------------------
# Configuration
# -----------------------------

SOURCE_DIR = Path("data/raw/NEU-CLS")
OUTPUT_DIR = Path("data")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# NEU-CLS filename prefixes
CLASS_MAP = {
    "Cr": "Crazing",
    "In": "Inclusion",
    "Pa": "Patches",
    "PS": "Pitted_Surface",
    "RS": "Rolled_In_Scale",
    "Sc": "Scratches"
}


# -----------------------------
# Create folders
# -----------------------------

for split in ["train", "validation", "test"]:
    for class_name in CLASS_MAP.values():
        folder = OUTPUT_DIR / split / class_name
        folder.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Read images
# -----------------------------

images_by_class = {name: [] for name in CLASS_MAP.values()}

for image_path in SOURCE_DIR.glob("*.bmp"):

    prefix = image_path.stem.split("_")[0]

    if prefix in CLASS_MAP:
        class_name = CLASS_MAP[prefix]
        images_by_class[class_name].append(image_path)


# -----------------------------
# Split dataset
# -----------------------------

print("\nNEU-CLS DATASET PREPARATION")
print("=" * 50)

total_images = 0

for class_name, images in images_by_class.items():

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    splits = {
        "train": train_images,
        "validation": val_images,
        "test": test_images
    }

    for split, split_images in splits.items():

        destination = OUTPUT_DIR / split / class_name

        for image_path in split_images:
            shutil.copy2(
                image_path,
                destination / image_path.name
            )

    print(
        f"{class_name:20s} "
        f"Total: {total:3d} | "
        f"Train: {len(train_images):3d} | "
        f"Val: {len(val_images):3d} | "
        f"Test: {len(test_images):3d}"
    )

    total_images += total


print("=" * 50)
print(f"Total images processed: {total_images}")
print("\nDataset preparation completed!")
