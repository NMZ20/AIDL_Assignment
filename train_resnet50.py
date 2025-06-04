import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

print(f"TensorFlow version: {tf.__version__}")

# Step 1: Load training dataset for ResNet50 (224x224)
print("Loading training dataset...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    'dataset/train/',
    image_size=(224, 224),  # ResNet50 standard size
    batch_size=32,
    label_mode='binary'
)

# Step 2: Load validation dataset
print("Loading validation dataset...")
val_dataset = tf.keras.utils.image_dataset_from_directory(
    'dataset/val/',
    image_size=(224, 224),
    batch_size=32,
    label_mode='binary'
)

# Step 3: Data preprocessing (normalization)
rescaling = tf.keras.layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (rescaling(x), y))
val_dataset = val_dataset.map(lambda x, y: (rescaling(x), y))

# Step 4: Create ResNet50 model
print("Creating ResNet50 model...")
base_model = ResNet50(
    weights='imagenet',  # Use pre-trained weights
    include_top=False,   # Don't include final classification layer
    input_shape=(224, 224, 3)
)

# Freeze the base model (transfer learning)
base_model.trainable = False

# Add custom classification layers
model = tf.keras.Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Step 5: Compile the model
print("Compiling model...")
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Print model summary
print("Model architecture:")
model.summary()

# Step 6: Train the model
print("Starting training...")
history = model.fit(
    train_dataset,
    epochs=5,  # Start with 5 epochs
    validation_data=val_dataset,
    verbose=1
)

# Step 7: Save the trained model
print("Saving model...")
model.save('resnet50_glasses_classifier.keras')
print("Model saved as 'resnet50_glasses_classifier.keras'")

# Step 8: Print training results
print(f"\nFinal Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
