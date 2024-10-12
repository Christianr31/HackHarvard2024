import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

# Normalize features
train_features = train_features / np.max(train_features)
test_features = test_features / np.max(test_features)

# Expand dimensions to match the input shape of the pretrained model
train_features = np.expand_dims(train_features, axis=-1)
test_features = np.expand_dims(test_features, axis=-1)

# Convert single-channel spectrograms to 3-channel by repeating the single channel
train_features = np.repeat(train_features, 3, axis=-1)
test_features = np.repeat(test_features, 3, axis=-1)

# Create the base model from the pre-trained model MobileNetV2
base_model = MobileNetV2(input_shape=(200, 513, 3), include_top=False, weights='imagenet')
base_model.trainable = False

# Add custom layers on top of the base model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(label_map), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_features, train_labels, epochs=10, validation_data=(test_features, test_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_features, test_labels, verbose=2)
print(f'\nTest accuracy: {test_acc}')