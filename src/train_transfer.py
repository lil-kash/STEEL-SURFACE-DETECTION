import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = (200, 200)
BATCH_SIZE = 32
EPOCHS = 15
NUM_CLASSES = 6

TRAIN_DIR = "data/train"
VAL_DIR = "data/validation"
TEST_DIR = "data/test"

MODEL_DIR = "models"
RESULTS_DIR = "results/plots"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

tf.random.set_seed(42)
np.random.seed(42)


# ============================================================
# DATA GENERATORS
# ============================================================

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

print("\nClass mapping:")
print(train_generator.class_indices)


# ============================================================
# PRETRAINED RESNET50
# ============================================================

base_model = ResNet50(
    weights="models/resnet50_weights.h5",
    include_top=False,
    input_shape=(200, 200, 3)
)

# Freeze pretrained layers
base_model.trainable = False


# ============================================================
# TRANSFER LEARNING MODEL
# ============================================================

inputs = layers.Input(
    shape=(200, 200, 3)
)

x = base_model(
    inputs,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.30)(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(0.20)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)

model = models.Model(
    inputs,
    outputs
)


# ============================================================
# MODEL SUMMARY
# ============================================================

model.summary()


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


# ============================================================
# CALLBACKS
# ============================================================

callbacks = [

    EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),

    ModelCheckpoint(
        "models/resnet50_transfer.keras",
        monitor="val_accuracy",
        save_best_only=True
    )
]


# ============================================================
# TRAIN
# ============================================================

print("\nStarting ResNet50 Transfer Learning...\n")

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
    "results/metrics/resnet50_history.json",
    "w"
) as f:
    json.dump(history_dict, f, indent=4)


# ============================================================
# ACCURACY GRAPH
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
plt.title("ResNet50 Transfer Learning - Accuracy")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "results/plots/resnet50_accuracy.png",
    dpi=300
)

plt.close()


# ============================================================
# LOSS GRAPH
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
plt.title("ResNet50 Transfer Learning - Loss")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    "results/plots/resnet50_loss.png",
    dpi=300
)

plt.close()


# ============================================================
# TEST EVALUATION
# ============================================================

print("\nEvaluating ResNet50 on test set...\n")

test_loss, test_accuracy = model.evaluate(
    test_generator
)

print(
    f"ResNet50 Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"ResNet50 Test Loss: "
    f"{test_loss:.4f}"
)


# ============================================================
# SAVE MODEL
# ============================================================

model.save(
    "models/resnet50_transfer_final.keras"
)

print("\n==========================================")
print("RESNET50 TRANSFER LEARNING COMPLETED")
print("==========================================")

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print("Model saved:")
print("models/resnet50_transfer_final.keras")
