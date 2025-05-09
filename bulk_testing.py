from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report

# Load the test dataset
test_dataset = tf.keras.utils.image_dataset_from_directory(
    'dataset/val/',
    image_size=(256, 256),  
    batch_size=32,
    label_mode='binary',  
    shuffle=False  
)

# Extract true labels
y_true = []
for _, labels in test_dataset:
    y_true.extend(labels.numpy())
y_true = np.array(y_true)

# Load the model I trained
custom_model = tf.keras.models.load_model('glasses_classifier_model.keras') 
# Load the pre-trained model       
pretrained_model = tf.keras.models.load_model('pretrained.keras') 

# Predict with both models
y_pred_custom = custom_model.predict(test_dataset)
y_pred_pretrained = pretrained_model.predict(test_dataset)

# Convert probabilities to binary labels
y_pred_custom_labels = (y_pred_custom > 0.5).astype(int).flatten()
y_pred_pretrained_labels = (y_pred_pretrained > 0.5).astype(int).flatten()

class_names = test_dataset.class_names  # e.g., ['with_glasses', 'without_glasses']

# Generate classification reports
print("\n--- Custom CNN Report ---")
print(classification_report(y_true, y_pred_custom_labels, target_names=class_names))

print("\n--- Pre-trained CNN Report ---")
print(classification_report(y_true, y_pred_pretrained_labels, target_names=class_names))


# # Evaluate performance
# accuracy_custom = accuracy_score(y_true, y_pred_custom_labels)
# f1_custom = f1_score(y_true, y_pred_custom_labels)

# accuracy_pretrained = accuracy_score(y_true, y_pred_pretrained_labels)
# f1_pretrained = f1_score(y_true, y_pred_pretrained_labels)

# # Print results in table format
# print("\nEvaluation Results:")
# print(f"{'Model':<25}{'Accuracy (%)':<15}{'F1-Score (%)'}")
# print(f"{'Custom CNN':<25}{accuracy_custom*100:<15.2f}{f1_custom*100:.2f}")
# print(f"{'Pre-trained CNN':<25}{accuracy_pretrained*100:<15.2f}{f1_pretrained*100:.2f}")