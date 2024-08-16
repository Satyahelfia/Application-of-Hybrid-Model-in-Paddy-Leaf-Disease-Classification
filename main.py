import torch
from models.cnn_model import CNNModel
from models.elm_model import ELM
from models.svm_model import build_svm
from utils.data import load_data
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
train_loader, test_loader = load_data('dataset1/train_images', 'dataset1/test_images')

# Build CNN Model
cnn_model = CNNModel()
cnn_model.eval()  # Set the model to evaluation mode

# Extract Features with CNN
features, labels = [], []
for images, targets in train_loader:
    outputs = cnn_model(images)
    features.append(outputs.detach().numpy())
    labels.append(targets.numpy())

features = np.vstack(features)
labels = np.hstack(labels)

# Prepare ELM Model
elm_model = ELM(input_size=features.shape[1], hidden_size=1000, output_size=10)
elm_model.fit(features, labels)

# Get ELM Output and Train SVM
svm_model = build_svm()
elm_features = elm_model.predict(features)

# Pastikan `elm_features` berbentuk 2D
if elm_features.ndim == 1:
    elm_features = elm_features.reshape(-1, 1)
svm_model.fit(elm_features, labels)

# Testing Phase
test_features, test_labels = [], []
for images, targets in test_loader:
    outputs = cnn_model(images)
    test_features.append(outputs.detach().numpy())
    test_labels.append(targets.numpy())

test_features = np.vstack(test_features)
test_labels = np.hstack(test_labels)

elm_test_output = elm_model.predict(test_features)

# Mengubah `elm_test_output` ke 2D
if elm_test_output.ndim == 1:
    elm_test_output = elm_test_output.reshape(-1, 1)
predictions = svm_model.predict(elm_test_output)

# Evaluasi hasil
accuracy = np.mean(predictions == test_labels)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Confusion Matrix
cm = confusion_matrix(test_labels, predictions)
print(f'Confusion Matrix:\n{cm}')

# Plot Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(test_labels), yticklabels=np.unique(test_labels))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()