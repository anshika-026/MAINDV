import tensorflow as tf
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix


# ==========================================
# 1. SETTINGS
# ==========================================

test_path = "dataset/test"

image_size = (48, 48)
batch_size = 32


# ==========================================
# 2. LOAD TEST DATASET
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
# 3. NORMALIZE TEST DATA
# ==========================================

normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

test_dataset = test_dataset.map(
    lambda images, labels: (
        normalization_layer(images),
        labels
    )
)


# ==========================================
# 4. LOAD MODEL V2
# ==========================================

model = tf.keras.models.load_model(
    "models/emotion_model_v2.keras"
)

print("\nModel V2 loaded successfully!")


# ==========================================
# 5. TEST MODEL
# ==========================================

test_loss, test_accuracy = model.evaluate(
    test_dataset
)


print("\n==============================")
print("MODEL V2 TEST RESULTS")
print("==============================")

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# ==========================================
# 6. GET PREDICTIONS
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

    y_true.extend(labels.numpy())

    y_pred.extend(predicted_labels)


y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ==========================================
# 7. CLASSIFICATION REPORT
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
# 8. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_true,
    y_pred
)


print("\n==============================")
print("CONFUSION MATRIX")
print("==============================\n")

print(cm)