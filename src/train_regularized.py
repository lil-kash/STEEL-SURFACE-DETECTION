import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = (200, 200)
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 6

TRAIN_DIR = "data/train"
VAL_DIR = "data/validation"
TEST_DIR = "data/test"

os.makedirs("models", exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)

tf.random.set_seed(42)
np.random.seed(42)

# ============================================================
# DATA AUGMENTATION
# ============================================================

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=8,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True
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
# REGULARIZED CNN
# ============================================================

l2_strength = 0.00001

model = models.Sequential([

    layers.Input(shape=(200, 200, 1)),

    # Block 1
    layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        kernel_regularizer=regularizers.l2(l2_strength)
    ),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D((2, 2)),

    # Block 2
    layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        kernel_regularizer=regularizers.l2(l2_strength)
    ),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D((2, 2)),

    # Block 3
    layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        kernel_regularizer=regularizers.l2(l2_strength)
    ),
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.MaxPooling2D((2, 2)),

    # Classification
    layers.GlobalAveragePooling2D(),

    layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_strength)
    ),

    layers.Dropout(0.40),

    layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])

# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

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
        "models/regularized_cnn.keras",
        monitor="val_accuracy",
        save_best_only=True
    )
]

# ============================================================
# TRAIN
# ============================================================

print("\nStarting Regularized CNN training...\n")

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ============================================================
# SAVE HISTORY
# ============================================================

history_dict = {
    key: [float(v) for v in values]
    for key, values in history.history.items()
}

with open(
    "results/metrics/regularized_history.json",
    "w"
) as f:
    json.dump(history_dict, f, indent=4)

# ============================================================
# ACCURACY PLOT
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
plt.title("Regularized CNN - Accuracy")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "results/plots/regularized_accuracy.png",
    dpi=300
)

plt.close()

# ============================================================
# LOSS PLOT
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
plt.title("Regularized CNN - Loss")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "results/plots/regularized_loss.png",
    dpi=300
)

plt.close()

# ============================================================
# TEST
# ============================================================

print("\nEvaluating Regularized CNN...\n")

test_loss, test_accuracy = model.evaluate(
    test_generator
)

print(
    f"Regularized Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Regularized Test Loss: "
    f"{test_loss:.4f}"
)

# ============================================================
# SAVE MODEL
# ============================================================

model.save(
    "models/regularized_cnn_final.keras"
)

print("\n==========================================")
print("REGULARIZED CNN TRAINING COMPLETED")
print("==========================================")

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print("Model saved:")
print("models/regularized_cnn_final.keras")
