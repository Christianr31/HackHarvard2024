import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# Load your data
train_features = np.load('model/data/train_features.npy')
train_labels = np.load('model/data/train_labels.npy', allow_pickle=True)
train_labels = [label.decode('utf-8') for label in train_labels]

test_features = np.load('model/data/test_features.npy')
test_labels = np.load('model/data/test_labels.npy', allow_pickle=True)
test_labels = [label.decode('utf-8') for label in test_labels]

# Convert labels to categorical
label_map = {label: idx for idx, label in enumerate(set(train_labels))}
train_labels = np.array([label_map[label] for label in train_labels])
test_labels = np.array([label_map[label] for label in test_labels])

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

# Assuming train_features, train_labels, test_features, and test_labels are already loaded and preprocessed

# Normalize features
train_features = train_features / np.max(train_features)
test_features = test_features / np.max(test_features)

# Verify the shape of the features
print(f"Original train features shape: {train_features.shape}")
print(f"Original test features shape: {test_features.shape}")

# Ensure the features have the correct shape (200, 513)
if train_features.shape[1:] != (200, 513):
    raise ValueError(f"Expected train features shape (200, 513), but got {train_features.shape[1:]}")
if test_features.shape[1:] != (200, 513):
    raise ValueError(f"Expected test features shape (200, 513), but got {test_features.shape[1:]}")

# Expand dimensions to match the input shape of the pretrained model
train_features = np.expand_dims(train_features, axis=-1)
test_features = np.expand_dims(test_features, axis=-1)

# Convert single-channel spectrograms to 3-channel by repeating the single channel
train_features = np.repeat(train_features, 3, axis=-1)
test_features = np.repeat(test_features, 3, axis=-1)

# Verify the new shape of the features
print(f"New train features shape: {train_features.shape}")
print(f"New test features shape: {test_features.shape}")

# Create the base model from the pre-trained model MobileNetV2
base_model = MobileNetV2(input_shape=(200, 513, 3), include_top=False, weights='imagenet')
base_model.trainable = False

# Add custom layers on top of the base model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(len(label_map), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_features, train_labels, epochs=20, validation_data=(test_features, test_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_features, test_labels, verbose=2)
print(f'Test accuracy: {test_acc:.2f}')
# Save the model
model.save('my_model.keras')

from collections import Counter
print(Counter(train_labels))


# 0 = soft, 1 = normal, 2 = loud, 3 = very loud