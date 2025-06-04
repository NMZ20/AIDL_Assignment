from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report

# Load the test dataset for custom model (256x256)
test_dataset_custom = tf.keras.utils.image_dataset_from_directory(
    'dataset/val/',
    image_size=(256, 256),  
    batch_size=32,
    label_mode='binary',  
    shuffle=False  
)

# Load the test dataset for ResNet50 (224x224)
test_dataset_resnet = tf.keras.utils.image_dataset_from_directory(
    'dataset/val/',
    image_size=(224, 224),  
    batch_size=32,
    label_mode='binary',  
    shuffle=False  
)

# Extract true labels for both datasets
y_true_custom = []
for _, labels in test_dataset_custom:
    y_true_custom.extend(labels.numpy())
y_true_custom = np.array(y_true_custom)

y_true_resnet = []
for _, labels in test_dataset_resnet:
    y_true_resnet.extend(labels.numpy())
y_true_resnet = np.array(y_true_resnet)

# Load both models
print("Loading custom CNN model...")
custom_model = tf.keras.models.load_model('glasses_classifier_model.keras') 

print("Loading ResNet50 model...")
resnet_model = tf.keras.models.load_model('resnet50_glasses_classifier.keras')

# Make predictions with both models
print("Making predictions with custom CNN...")
y_pred_custom = custom_model.predict(test_dataset_custom)
y_pred_custom_labels = (y_pred_custom > 0.5).astype(int).flatten()

print("Making predictions with ResNet50...")
y_pred_resnet = resnet_model.predict(test_dataset_resnet)
y_pred_resnet_labels = (y_pred_resnet > 0.5).astype(int).flatten()

class_names = test_dataset_custom.class_names

# Generate classification reports
print("\n" + "="*60)
print("CUSTOM CNN REPORT")
print("="*60)
print(classification_report(y_true_custom, y_pred_custom_labels, target_names=class_names))

print("\n" + "="*60)
print("RESNET50 TRANSFER LEARNING REPORT")
print("="*60)
print(classification_report(y_true_resnet, y_pred_resnet_labels, target_names=class_names))

# Calculate and display accuracy comparison
custom_accuracy = accuracy_score(y_true_custom, y_pred_custom_labels)
resnet_accuracy = accuracy_score(y_true_resnet, y_pred_resnet_labels)

print("\n" + "="*60)
print("PERFORMANCE COMPARISON")
print("="*60)
print(f"Custom CNN Accuracy:    {custom_accuracy:.4f} ({custom_accuracy*100:.2f}%)")
print(f"ResNet50 Accuracy:      {resnet_accuracy:.4f} ({resnet_accuracy*100:.2f}%)")
print(f"Improvement:            {'+' if resnet_accuracy > custom_accuracy else ''}{(resnet_accuracy-custom_accuracy)*100:.2f}%")
