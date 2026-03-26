# This is what I have for the ai image comparison so far, im just using 9 real and 9 ai images, use some for
# training, then some for testing (grabbed them manually from the web and my phone,
# they are relatively obvious though). Need a folder called images with 2 folders inside called "real" and "ai"
# filled with images AND also have to download some stuff through ur terminal.
# python -m pip install opencv-python numpy scikit-learn (put in terminal)
# make sure whatever IDE ur using is the "main" one or default one (for thonny, you have to go into tools
# then options, then interpreter and go down to the python executable and choose the correct one).
# last step only necessary if you are on thonny AND your cv2 import doesnt work even though it is installed.
# AI Description:
'''
Program Description:
This program is a small proof-of-concept image classification system that attempts to
distinguish between real images and clearly AI-generated images. It is designed as an
early baseline for a larger research project involving quantum computing and image
classification. At this stage, the program uses classical image preprocessing and a
classical machine learning model so that the image pipeline can be tested and verified
before moving to a quantum-emulator-based approach.

What the program does:
1. It finds the folder where this Python script is stored.
2. It builds paths to two image folders:
      - images/real
      - images/ai
3. It loads every valid image file from those folders.
4. Each image is preprocessed in the same way:
      - read from disk with OpenCV
      - converted to grayscale
      - resized to 32 x 32 pixels
      - normalized so pixel values are between 0 and 1
5. The program stores all processed images in a data array.
6. It assigns labels to each image:
      - 0 = real image
      - 1 = AI-generated image
7. Since machine learning models usually need 1-dimensional input, each 32 x 32 image
   is flattened into a vector of length 1024.
8. The full dataset is split into training data and testing data using a stratified
   train/test split so both classes remain balanced.
9. A Logistic Regression classifier is trained on the training data. This serves as a
   classical baseline model.
10. The trained model predicts the labels of the test images.
11. The program prints:
      - cross-validation scores
      - average cross-validation accuracy
      - predicted labels
      - actual labels
      - test accuracy

What the output shows:
- "Cross-validation scores" shows how well the classifier performed across multiple
  different splits of the dataset.
- "Average accuracy" shows the mean performance across those cross-validation runs.
- "Predictions" shows what the model guessed for each test image.
- "Actual" shows the true labels for those same test images.
- "Accuracy" shows how many of the test images were classified correctly in that
  specific train/test split.
'''

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

base_dir = os.path.dirname(os.path.abspath(__file__))
real_folder = os.path.join(base_dir, "images", "real")
ai_folder = os.path.join(base_dir, "images", "ai")

def preprocess_image(path, size=(32, 32)):
    img = cv2.imread(path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, size)
    gray = gray.astype(np.float32) / 255.0
    return gray

data = []
labels = []

valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

for filename in os.listdir(real_folder):
    if not filename.lower().endswith(valid_ext):
        continue

    path = os.path.join(real_folder, filename)
    img = preprocess_image(path)
    data.append(img)
    labels.append(0)

for filename in os.listdir(ai_folder):
    if not filename.lower().endswith(valid_ext):
        continue

    path = os.path.join(ai_folder, filename)
    img = preprocess_image(path)
    data.append(img)
    labels.append(1)

data = np.array(data)
labels = np.array(labels)

#print("Data shape:", data.shape)
#print("Labels:", labels)

#print("X shape:", X.shape)
#print("y shape:", y.shape)
X = data.reshape(len(data), -1)
y = labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

#print("Train size:", len(X_train))
#print("Test size:", len(X_test))
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
scores = cross_val_score(model, X, y, cv=3)

predictions = model.predict(X_test)

print("Cross-validation scores:", scores)
print("Average accuracy:", scores.mean())
print("Predictions:", predictions)
print("Actual:     ", y_test)
print("Accuracy:", accuracy_score(y_test, predictions))