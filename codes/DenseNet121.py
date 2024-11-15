import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

# Mount Google Drive if not already done
from google.colab import drive
drive.mount("/content/drive")

# Define the paths to your data folders
data_dir_with_tumor = "/content/drive/MyDrive/LOP_3-2/datasets/yes"
data_dir_without_tumor = "/content/drive/MyDrive/LOP_3-2/datasets/no"

# Initialize empty lists to store data and labels
data = []
labels = []  # Use binary labels (0 for with tumor, 1 for without tumor)

# Define a common size for all images
common_size = (128, 128)

# Load images with tumors
for filename in os.listdir(data_dir_with_tumor):
    if filename.endswith(".jpg"):
        image_path = os.path.join(data_dir_with_tumor, filename)
        # Load and preprocess the image (resize, convert to grayscale, normalize, etc.) as needed
        image = Image.open(image_path)
        image = image.resize(common_size)
        image = np.array(image.convert("RGB")) / 255.0  # Convert to RGB and normalize pixel values
        data.append(image)
        labels.append(0)  # Label 0 for with tumor

# Load images without tumors
for filename in os.listdir(data_dir_without_tumor):
    if filename.endswith(".jpg"):
        image_path = os.path.join(data_dir_without_tumor, filename)
        # Load and preprocess the image (resize, convert to grayscale, normalize, etc.) as needed
        image = Image.open(image_path)
        image = image.resize(common_size)
        image = np.array(image.convert("RGB")) / 255.0  # Convert to RGB and normalize pixel values
        data.append(image)
        labels.append(1)  # Label 1 for without tumor

# Convert data and labels to NumPy arrays
data = np.array(data)
labels = np.array(labels)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Load the pre-trained DenseNet121 model without the top classification layers
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Add custom classification layers on top of the base model
model = Sequential([
    base_model,
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Freeze the layers of the pre-trained base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history=model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=30)


# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Loss: {loss:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')

# Predict on the test set
y_pred = model.predict(x_test)
y_pred_binary = np.round(y_pred)

# Calculate Precision, Recall, and F1 Score
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

y_pred = model.predict(x_test)
y_pred_binary = np.round(y_pred)

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_binary)

# Plot the confusion matrix
class_names = ['With Tumor', 'Without Tumor']
df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
plt.figure(figsize=(6, 4))
sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for DenseNet121')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy vs. Epochs for Densenet 121')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
