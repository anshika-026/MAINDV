# Real-Time Facial Expression Recognition

A real-time facial expression recognition system that uses **Deep Learning and Computer Vision** to detect faces from a live webcam feed and classify facial expressions into seven emotion categories.

The project uses a **Convolutional Neural Network (CNN)** built with **TensorFlow/Keras** for emotion classification and **OpenCV** for real-time face detection.

## Features

* Real-time facial expression recognition using a webcam
* Face detection using OpenCV Haar Cascade
* CNN-based emotion classification
* Recognition of seven facial expressions:

  * Angry
  * Disgust
  * Fear
  * Happy
  * Neutral
  * Sad
  * Surprise
* Image preprocessing and normalization
* Data augmentation during training
* Batch normalization and dropout
* Handling of class imbalance
* Prediction smoothing for stable real-time results
* Confidence score displayed with the predicted emotion

---
## Project Workflow

```text
Webcam
   ↓
Capture Video Frame
   ↓
Face Detection
   ↓
Face Cropping
   ↓
Convert to Grayscale
   ↓
Resize to 48 × 48
   ↓
Normalize Pixel Values
   ↓
CNN Model
   ↓
Emotion Prediction
   ↓
Prediction Smoothing
   ↓
Display Emotion and Confidence
```

---

## Technologies Used

* Python
* TensorFlow
* Keras
* OpenCV
* NumPy
* Scikit-learn

---

## Dataset

The dataset contains facial images belonging to seven emotion categories:

* Angry
* Disgust
* Fear
* Happy
* Neutral
* Sad
* Surprise

The dataset is divided into:

* Training Set: 80%
* Validation Set: 10%
* Testing Set: 10%

The images are grayscale images with a size of:

```text
48 × 48 pixels
```

---

## Dataset Distribution

### Training Dataset

| Emotion  | Number of Images |
| -------- | ---------------: |
| Angry    |             3995 |
| Disgust  |              436 |
| Fear     |             4097 |
| Happy    |             7215 |
| Neutral  |             4965 |
| Sad      |             4830 |
| Surprise |             3171 |

The dataset is imbalanced, particularly for the **Disgust** class. Custom class weighting was used in Model Version 3 to improve recognition of underrepresented emotions.

---

## Project Structure

```text
real-time-facial-expression/
│
├── dataset/
│   ├── train/
│   ├── val/
│   └── test/
│
├── models/
│   ├── emotion_model.keras
│   ├── emotion_model_v2.keras
│   └── emotion_model_v3.keras
│
├── src/
│   ├── build_model.py
│   ├── explore_dataset.py
│   ├── face_detection.py
│   ├── real_time_emotion.py
│   │
│   ├── train_model.py
│   ├── evaluate_model.py
│   │
│   ├── train_model_v2.py
│   ├── evaluate_model_v2.py
│   │
│   ├── train_model_v3.py
│   ├── evaluate_model_v3.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Model Development

Multiple versions of the CNN model were trained and evaluated.

## Model Version 1

The first CNN model was trained on the dataset.

### Results

* Test Accuracy: **48.76%**

The model showed difficulty in recognizing several emotion classes.

---

## Model Version 2

The architecture was improved using:

* Data augmentation
* Batch normalization
* Dropout
* Improved CNN architecture

### Results

* Test Accuracy: **60%**

The model showed significant improvement in overall accuracy. However, it failed to predict the **Disgust** class effectively.

---

## Model Version 3

The final model was developed by keeping the improved architecture from Version 2 and applying moderate custom class weighting.

### Improvements

* Data augmentation
* Batch normalization
* Dropout
* Early stopping
* Learning rate reduction
* Custom class weights

---

# Final Model Performance

The final Model V3 achieved:

```text
Test Accuracy: 60.82%
Test Loss: 1.0293
```

## Classification Performance

| Emotion  | Precision | Recall | F1-Score |
| -------- | --------: | -----: | -------: |
| Angry    |      0.53 |   0.53 |     0.53 |
| Disgust  |      0.45 |   0.24 |     0.31 |
| Fear     |      0.47 |   0.26 |     0.33 |
| Happy    |      0.82 |   0.87 |     0.84 |
| Neutral  |      0.50 |   0.76 |     0.61 |
| Sad      |      0.47 |   0.37 |     0.41 |
| Surprise |      0.74 |   0.75 |     0.74 |

### Overall Performance

```text
Accuracy: 60.82%
Weighted F1-Score: 0.59
```

The model performs particularly well for:

* Happy
* Surprise
* Neutral

The model has comparatively lower performance for:

* Disgust
* Fear
* Sad

This is mainly influenced by class imbalance and the difficulty of distinguishing subtle facial expressions.

---

# Model Architecture

The final model uses a Convolutional Neural Network (CNN).

The architecture consists of:

```text
Input Layer
    ↓
Data Augmentation
    ↓
Conv2D + Batch Normalization
    ↓
Conv2D
    ↓
MaxPooling
    ↓
Dropout
    ↓
Conv2D + Batch Normalization
    ↓
Conv2D
    ↓
MaxPooling
    ↓
Dropout
    ↓
Conv2D + Batch Normalization
    ↓
Conv2D
    ↓
MaxPooling
    ↓
Dropout
    ↓
Global Average Pooling
    ↓
Dense Layer
    ↓
Dropout
    ↓
Softmax Output Layer
```

The output layer contains seven neurons corresponding to the seven emotion classes.

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/BETTER678/real-time-facial-expression-recognition.git
```

Navigate to the project directory:

```bash
cd real-time-facial-expression
```

---

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the virtual environment.

### Windows

```bash
venv\Scripts\activate
```

### Linux/macOS

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Alternatively:

```bash
pip install tensorflow opencv-python numpy scikit-learn
```

---

# Training the Model

To train Model Version 3:

```bash
python src/train_model_v3.py
```

The trained model will be saved in:

```text
models/emotion_model_v3.keras
```

---

# Evaluating the Model

To evaluate Model V3 on the test dataset:

```bash
python src/evaluate_model_v3.py
```

The evaluation includes:

* Test Accuracy
* Test Loss
* Precision
* Recall
* F1-Score
* Confusion Matrix

The model is evaluated on the test dataset, which was not used for training.

---

# Running Face Detection

To test face detection:

```bash
python src/face_detection.py
```

The webcam will open and a rectangle will appear around detected faces.

Press:

```text
Q
```

to close the application.

---

# Running Real-Time Emotion Recognition

Run:

```bash
python src/real_time_emotion.py
```

The application will:

1. Open the webcam.
2. Detect the face.
3. Extract the facial region.
4. Convert the image to grayscale.
5. Resize it to 48 × 48.
6. Normalize pixel values.
7. Send the image to the trained CNN model.
8. Predict the facial expression.
9. Display the predicted emotion and confidence score.

Press:

```text
Q
```

to exit the application.

---

# Real-Time Prediction Smoothing

The application uses prediction smoothing to reduce rapid changes in the predicted emotion.

The predictions from the last 10 frames are stored and averaged.

```text
Frame 1 → Happy
Frame 2 → Happy
Frame 3 → Neutral
Frame 4 → Happy
Frame 5 → Happy

        ↓

Average Prediction

        ↓

Happy
```

This improves the stability of the real-time output.

---

# Image Preprocessing

Before an image is sent to the CNN model, the following preprocessing steps are applied:

```text
Detected Face
     ↓
Crop Face
     ↓
Convert to Grayscale
     ↓
Resize to 48 × 48
     ↓
Convert to Float32
     ↓
Normalize Pixel Values (0–1)
     ↓
Add Channel Dimension
     ↓
Add Batch Dimension
```

The final input shape is:

```text
(1, 48, 48, 1)
```

Where:

* `1` = Batch size
* `48` = Image height
* `48` = Image width
* `1` = Grayscale channel

---

# Challenges Faced

## Class Imbalance

The dataset contains significantly fewer images for the **Disgust** emotion compared to other classes.

For example:

```text
Happy   → 7215 images
Disgust → 436 images
```

This caused difficulty in training the model to recognize Disgust correctly.

Custom class weighting was used in Version 3 to improve the performance of the underrepresented class.

---

## Similar Facial Expressions

Some emotions share similar facial characteristics.

For example:

* Fear can be confused with Angry or Sad.
* Sad can be confused with Neutral.
* Neutral expressions can be confused with other subtle emotions.

These similarities affect classification performance.

---

## Real-Time Webcam Differences

The training dataset and webcam input may differ in:

* Lighting
* Camera angle
* Face position
* Facial appearance
* Background

This can affect real-world performance.

---

# Future Improvements

Possible improvements include:

* Using a larger and more balanced dataset
* Collecting additional Disgust samples
* Using transfer learning
* Using a pre-trained facial recognition model
* Implementing a stronger face detector
* Improving recognition of subtle emotions
* Adding a graphical user interface
* Displaying FPS
* Supporting multiple face tracking
* Deploying the application as a web application

---

# Key Learning Outcomes

Through this project, I learned:

* Convolutional Neural Networks
* Image preprocessing
* Deep Learning with TensorFlow and Keras
* Model training and evaluation
* Training, validation, and test datasets
* Data augmentation
* Batch normalization
* Dropout
* Class imbalance handling
* Precision, Recall, and F1-score
* Confusion matrices
* OpenCV
* Face detection
* Real-time webcam processing
* Integrating a trained deep learning model into a real-time application

---

# Conclusion

This project demonstrates an end-to-end implementation of a real-time facial expression recognition system using Deep Learning and Computer Vision.

A CNN model was trained to classify seven facial expressions and integrated with OpenCV to perform real-time emotion prediction through a webcam.

The final Model V3 achieved a test accuracy of **60.82%** and showed strong performance in recognizing expressions such as **Happy** and **Surprise**.

The project provides practical experience in building, training, evaluating, and deploying a machine learning model in a real-time application.

---

# Author

**Srushti Khillare**

B.Tech in Information Technology

---

## If you found this project useful

Consider giving the repository a star!
