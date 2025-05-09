import numpy as np
import tensorflow as tf

# Create the training dataset
train_set = tf.keras.utils.image_dataset_from_directory(
    'dataset/train',
    image_size=(256, 256),
    batch_size=32,
    label_mode='binary'
)

# Data augmentation layer
train_data_augmentation = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),  # Rescale pixel values
    tf.keras.layers.RandomFlip('horizontal'),  # Apply horizontal flip
    tf.keras.layers.RandomZoom(0.2),  # Apply zoom augmentation
    tf.keras.layers.RandomRotation(0.2),  # Apply rotation for better variety
    tf.keras.layers.RandomShear(0.2)  # Apply shear transformation
])

# Apply the augmentation pipeline
augmented_train_dataset = train_set.map(lambda x, y: (train_data_augmentation(x, training=True), y))


# Create the test dataset
test_set = tf.keras.utils.image_dataset_from_directory(
    'dataset/test',
    image_size=(256, 256),
    batch_size=32,
    label_mode='binary',
    shuffle=False
)

# Data augmentation layer
test_data_augmentation = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255)
])

# Apply the augmentation pipeline
augmented_test_dataset = test_set.map(lambda x, y: (test_data_augmentation(x, training=False), y))
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[256, 256, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(128, activation='relu'))
cnn.add(tf.keras.layers.Dense(1, activation='sigmoid'))
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# Add early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=3,  # Stop if no improvement for 3 epochs
    min_delta=0.005,  # Minimum change to qualify as improvement
    restore_best_weights=True  # Restore model to best epoch
)

cnn.fit(x = augmented_train_dataset, validation_data = augmented_test_dataset, epochs = 5, callbacks=[early_stopping])

# Save the trained model
model_save_path = 'glasses_classifier_model.keras'
cnn.save(model_save_path)
print(f"Model saved to {model_save_path}")

# """## Part 4 - Making a single prediction"""

# test_image = tf.keras.utils.load_img(
#     'dataset/single_prediction/cat_or_dog_1.jpg',
#     target_size=(64, 64),
# )
# test_image = tf.keras.utils.img_to_array(test_image)
# test_image = np.expand_dims(test_image, axis = 0)
# result = cnn.predict(test_image)