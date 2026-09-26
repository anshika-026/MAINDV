import tensorflow as tf
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix


# ==========================================
# SETTINGS
# ==========================================

test_path = "dataset/test"

image_size = (48, 48)
batch_size = 32


# ==========================================
# LOAD TEST DATASET
# ==========================================

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=image_size,
    batch_size=batch_size,
    color_mode="grayscale",
    shuffle=False
)


# Get class names

class_names = test_dataset.class_names

print("\nClass Names:")
print(class_names)


# ==========================================
# NORMALIZE TEST IMAGES
# ==========================================

normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

test_dataset = test_dataset.map(
    lambda images, labels: (
        normalization_layer(images),
        labels
    )
)


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = tf.keras.models.load_model(
    "models/emotion_model.keras"
)

print("\nModel loaded successfully!")


# ==========================================
# EVALUATE MODEL
# ==========================================

test_loss, test_accuracy = model.evaluate(
    test_dataset
)


print("\n==============================")

print("TEST RESULTS")

print("==============================")

print("Test Loss:", test_loss)

print("Test Accuracy:", test_accuracy)


# ==========================================
# GET PREDICTIONS
# ==========================================

y_true = []
y_pred = []


for images, labels in test_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_labels = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(
        labels.numpy()
    )

    y_pred.extend(
        predicted_labels
    )


# Convert to NumPy arrays

y_true = np.array(y_true)

y_pred = np.array(y_pred)


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\n==============================")

print("CLASSIFICATION REPORT")

print("==============================\n")


print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_true,
    y_pred
)


print("\n==============================")

print("CONFUSION MATRIX")

print("==============================\n")


print(cm)