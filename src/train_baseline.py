import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = (200, 200)
BATCH_SIZE = 32
EPOCHS = 25
NUM_CLASSES = 6

TRAIN_DIR = "data/train"
VAL_DIR = "data/validation"
TEST_DIR = "data/test"

MODEL_DIR = "models"
RESULTS_DIR = "results/plots"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Reproducibility
tf.random.set_seed(42)
np.random.seed(42)


# ============================================================
# DATA LOADING
# ============================================================

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print("\nClass mapping:")
print(train_generator.class_indices)


# ============================================================
# BASELINE CNN
# ============================================================

model = models.Sequential([
    
    layers.Input(shape=(200, 200, 1)),

    # Block 1
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    ),
    layers.MaxPooling2D((2, 2)),

    # Block 2
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),
    layers.MaxPooling2D((2, 2)),

    # Block 3
    layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    ),
    layers.MaxPooling2D((2, 2)),

    # Classification
    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])


# ============================================================
# MODEL SUMMARY
# ============================================================

model.summary()


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),

    ModelCheckpoint(
        "models/baseline_cnn.keras",
        monitor="val_accuracy",
        save_best_only=True
    )
]


# ============================================================
# TRAINING
# ============================================================

print("\nStarting Baseline CNN training...\n")

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_dict = {
    key: [float(value) for value in values]
    for key, values in history.history.items()
}

with open(
    "results/metrics/baseline_history.json",
    "w"
) as file:
    json.dump(history_dict, file, indent=4)


# ============================================================
# ACCURACY CURVE
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Baseline CNN - Training vs Validation Accuracy")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "results/plots/baseline_accuracy.png",
    dpi=300
)

plt.close()


# ============================================================
# LOSS CURVE
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Baseline CNN - Training vs Validation Loss")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "results/plots/baseline_loss.png",
    dpi=300
)

plt.close()


# ============================================================
# TEST EVALUATION
# ============================================================

print("\nEvaluating on test set...\n")

test_loss, test_accuracy = model.evaluate(
    test_generator
)

print(
    f"Baseline Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Baseline Test Loss: "
    f"{test_loss:.4f}"
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(
    "models/baseline_cnn_final.keras"
)

print("\n===================================")
print("BASELINE CNN TRAINING COMPLETED")
print("===================================")

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print("Model saved to:")
print("models/baseline_cnn_final.keras")

print("\nPlots saved to:")
print("results/plots/")
