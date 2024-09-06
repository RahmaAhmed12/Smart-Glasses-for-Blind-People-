
import cv2
import os
import numpy as np
from sklearn.svm import SVC

correct_order = ['Ebtihal', 'Menna', 'Rahma Ahmed', 'Rahma Medhat', 'alaa', 'ansam']


def extract_features(image_path):
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image: {image_path}")
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Extract features for the first face detected
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (100, 100))
        # Flatten the face image
        face = face.flatten()
        return face
    else:
        print(f"No faces detected in image: {image_path}")
        return None


def extract_features_from_folder(base_folder_path):
    features = []
    labels = []
    class_names = []  # Keep track of class names

    # Debugging: Print folder names and their corresponding labels
    for label, folder in enumerate(correct_order):
        print(f"Folder: {folder}, Label: {label}")

    for label, folder in enumerate(correct_order):
        folder_path = os.path.join(base_folder_path, folder)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                image_path = os.path.join(folder_path, filename)
                feature = extract_features(image_path)
                if feature is not None:
                    features.append(feature)
                    labels.append(label)  # Use the correct label based on the order
                    class_names.append(folder)  # Add the class name to the list
        else:
            print(f"Folder does not exist: {folder_path}")
    return np.array(features), np.array(labels), class_names  # Return class names along with features and labels


# Train the model
base_folder_path = '/kaggle/input/reco-zft/face'
features, labels, class_names = extract_features_from_folder(base_folder_path)

# Check if we have any features before training
if features.size == 0:
    print("No features extracted. Check your data and paths.")
else:
    model = SVC(gamma='auto')
    model.fit(features, labels)

    # Predict the class for a single test image
    test_image_path = '/kaggle/input/reco-zft/face/ansam/20240424165215_000188.JPG'  # Change this path to your test image path
    test_feature = extract_features(test_image_path)

    if test_feature is None:
        print("No features extracted from test image. Exiting.")
    else:
        predicted_label = model.predict([test_feature])[0]

        # Correctly map the predicted label back to the class name
        predicted_class_name = correct_order[predicted_label]

        print(f"Predicted class: {predicted_class_name}")
        print(f"Predicted label: {predicted_label}")

import joblib

# Specify the file path where you want to save the model
model_path = '/kaggle/working/svm_model.h5'

# Save the trained model
joblib.dump(model, model_path)
print("Model saved successfully as SVM_model.h5.")