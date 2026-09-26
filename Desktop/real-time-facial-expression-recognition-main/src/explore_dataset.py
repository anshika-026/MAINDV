import os
import cv2
import matplotlib.pyplot as plt


# Path to training dataset
train_path = "dataset/train"


# Get emotion classes
emotions = os.listdir(train_path)

print("Emotion Classes:")
print(emotions)


# Count images
print("\nNumber of Training Images:")

for emotion in emotions:

    emotion_path = os.path.join(train_path, emotion)

    number_of_images = len(os.listdir(emotion_path))

    print(emotion, ":", number_of_images)


# Create a figure
plt.figure(figsize=(15, 5))


# Display one image from each emotion
for index, emotion in enumerate(emotions):

    # Get emotion folder path
    emotion_path = os.path.join(train_path, emotion)

    # Get first image
    image_name = os.listdir(emotion_path)[0]

    # Create complete image path
    image_path = os.path.join(
        emotion_path,
        image_name
    )

    # Read image
    image = cv2.imread(image_path)

    # Print image shape
    print("\nEmotion:", emotion)
    print("Image shape:", image.shape)


    # Create subplot
    plt.subplot(
        1,
        len(emotions),
        index + 1
    )


    # Check if image is color
    if len(image.shape) == 3:

        # Convert BGR to RGB
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        plt.imshow(image)

    else:

        # Display grayscale image
        plt.imshow(
            image,
            cmap="gray"
        )


    # Add title
    plt.title(emotion)

    # Remove axis
    plt.axis("off")


plt.show()