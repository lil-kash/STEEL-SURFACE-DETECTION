import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

MODEL_PATH = "models/baseline_cnn_final.keras"
TEST_DIR = "data/test"

IMG_SIZE = (200, 200)
BATCH_SIZE = 32

os.makedirs("results/metrics", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load test data
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# Predictions
predictions = model.predict(test_generator)

y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

class_names = list(test_generator.class_indices.keys())

# Classification report
report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)

print("\nBASELINE CNN CLASSIFICATION REPORT")
print("=" * 60)
print(report)

# Save report
with open(
    "results/metrics/baseline_classification_report.txt",
    "w"
) as f:
    f.write(report)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(9, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Baseline CNN - Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "results/plots/baseline_confusion_matrix.png",
    dpi=300
)

plt.close()

print("\nConfusion matrix saved.")

print(
    "\nEvaluation completed successfully!"
)
