import os
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ============================================================
# CONFIG
# ============================================================

IMG_SIZE = (200, 200)
BATCH_SIZE = 32
EPOCHS = 8
NUM_CLASSES = 6

TRAIN_DIR = "data/train"
VAL_DIR = "data/validation"
TEST_DIR = "data/test"

os.makedirs("results/metrics", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)

tf.random.set_seed(42)
np.random.seed(42)


# ============================================================
# DATA
# ============================================================

normal_train = ImageDataGenerator(
    rescale=1.0 / 255
)

augmented_train = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=8,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.05
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)


def get_generators(augmentation=False):

    train_gen = (
        augmented_train if augmentation else normal_train
    ).flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )

    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    return train_gen, val_gen, test_gen


# ============================================================
# MODEL BUILDER
# ============================================================

def build_model(mode):

    use_dropout = mode == "Dropout"
    use_batchnorm = mode == "BatchNorm"
    use_l2 = mode == "L2"

    l2_value = 0.00001

    model = models.Sequential()

    model.add(
        layers.Input(shape=(200, 200, 1))
    )

    # --------------------------------------------------------
    # BLOCK 1
    # --------------------------------------------------------

    if use_l2:
        model.add(
            layers.Conv2D(
                32,
                (3, 3),
                padding="same",
                kernel_regularizer=regularizers.l2(l2_value)
            )
        )
    else:
        model.add(
            layers.Conv2D(
                32,
                (3, 3),
                padding="same"
            )
        )

    if use_batchnorm:
        model.add(layers.BatchNormalization())

    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    # --------------------------------------------------------
    # BLOCK 2
    # --------------------------------------------------------

    if use_l2:
        model.add(
            layers.Conv2D(
                64,
                (3, 3),
                padding="same",
                kernel_regularizer=regularizers.l2(l2_value)
            )
        )
    else:
        model.add(
            layers.Conv2D(
                64,
                (3, 3),
                padding="same"
            )
        )

    if use_batchnorm:
        model.add(layers.BatchNormalization())

    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    # --------------------------------------------------------
    # BLOCK 3
    # --------------------------------------------------------

    if use_l2:
        model.add(
            layers.Conv2D(
                128,
                (3, 3),
                padding="same",
                kernel_regularizer=regularizers.l2(l2_value)
            )
        )
    else:
        model.add(
            layers.Conv2D(
                128,
                (3, 3),
                padding="same"
            )
        )

    if use_batchnorm:
        model.add(layers.BatchNormalization())

    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    # --------------------------------------------------------
    # CLASSIFIER
    # --------------------------------------------------------

    model.add(layers.Flatten())

    if use_l2:
        model.add(
            layers.Dense(
                128,
                activation="relu",
                kernel_regularizer=regularizers.l2(l2_value)
            )
        )
    else:
        model.add(
            layers.Dense(
                128,
                activation="relu"
            )
        )

    if use_dropout:
        model.add(layers.Dropout(0.30))

    model.add(
        layers.Dense(
            NUM_CLASSES,
            activation="softmax"
        )
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# EXPERIMENTS
# ============================================================

experiments = [
    "Dropout",
    "BatchNorm",
    "L2",
    "Augmentation"
]

results = []


for experiment in experiments:

    print("\n")
    print("=" * 60)
    print("EXPERIMENT:", experiment)
    print("=" * 60)

    augmentation = experiment == "Augmentation"

    train_gen, val_gen, test_gen = get_generators(
        augmentation
    )

    model = build_model(experiment)

    start_time = time.time()

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        verbose=1
    )

    training_time = time.time() - start_time

    test_loss, test_accuracy = model.evaluate(
        test_gen,
        verbose=0
    )

    best_val_accuracy = max(
        history.history["val_accuracy"]
    )

    result = {
        "Experiment": experiment,
        "Test Accuracy": test_accuracy,
        "Test Loss": test_loss,
        "Best Validation Accuracy": best_val_accuracy,
        "Training Time (sec)": training_time
    }

    results.append(result)

    print("\nRESULT")
    print("-" * 40)
    print(
        f"Test Accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )
    print(
        f"Test Loss: "
        f"{test_loss:.4f}"
    )
    print(
        f"Best Validation Accuracy: "
        f"{best_val_accuracy * 100:.2f}%"
    )
    print(
        f"Training Time: "
        f"{training_time:.2f} seconds"
    )


# ============================================================
# SAVE CSV
# ============================================================

csv_path = "results/metrics/ablation_results.csv"

with open(
    csv_path,
    "w",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "Experiment",
            "Test Accuracy",
            "Test Loss",
            "Best Validation Accuracy",
            "Training Time (sec)"
        ]
    )

    writer.writeheader()
    writer.writerows(results)


# ============================================================
# PRINT FINAL TABLE
# ============================================================

print("\n")
print("=" * 75)
print("REGULARIZATION ABLATION RESULTS")
print("=" * 75)

for r in results:

    print(
        f"{r['Experiment']:15}"
        f" | Test Acc: {r['Test Accuracy']*100:6.2f}%"
        f" | Val Acc: {r['Best Validation Accuracy']*100:6.2f}%"
        f" | Time: {r['Training Time (sec)']:7.1f}s"
    )


# ============================================================
# COMPARISON GRAPH
# ============================================================

names = [
    r["Experiment"]
    for r in results
]

accuracies = [
    r["Test Accuracy"] * 100
    for r in results
]

plt.figure(figsize=(10, 6))

bars = plt.bar(
    names,
    accuracies
)

plt.axhline(
    y=97.04,
    linestyle="--",
    label="Baseline CNN (97.04%)"
)

plt.xlabel("Regularization Technique")
plt.ylabel("Test Accuracy (%)")

plt.title(
    "Regularization Ablation Study"
)

plt.ylim(0, 105)

plt.legend()
plt.grid(
    axis="y",
    alpha=0.3
)

for bar, value in zip(
    bars,
    accuracies
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    "results/plots/ablation_accuracy.png",
    dpi=300
)

plt.close()

print("\nAblation study completed.")
print(
    f"Results saved to: {csv_path}"
)
