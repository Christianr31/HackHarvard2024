import json
import os
import shutil
import wave
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

from pydub import AudioSegment
# Utility function to get sound and frame rate info

def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = np.frombuffer(frames, 'int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate

work_dir = 'model/work'
if os.path.exists(work_dir):
    shutil.rmtree(work_dir)
os.makedirs(work_dir)

# def spectrogram_from_file(file_path):
#     sound_info, frame_rate = get_wav_info(file_path)
#     Pxx, freqs, bins, im = plt.specgram(sound_info, Fs=frame_rate)
#     plt.savefig(f'model/train/{label}/train_{file_stem}.png')
#     # count += 1
#     plt.close()
#     return
# Reload a fresh Keras model from the .keras zip archive
new_model = tf.keras.models.load_model('my_model.keras')
# Load class names from the JSON file
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

print("Loaded class names:", class_names)

# Show the model architecture
new_model.summary()

file_path = 'test.wav'
sound_info, frame_rate = get_wav_info(file_path)
Pxx, freqs, bins, im = plt.specgram(sound_info, Fs=frame_rate)
plt.savefig(f'model/work/test.png')
plt.close()

spectrogram_path = 'model/work/test.png'
spectrogram = tf.io.read_file(spectrogram_path)
spectrogram = tf.image.decode_image(spectrogram, channels=3)
spectrogram = tf.image.resize(spectrogram, [256, 256])
spectrogram = tf.expand_dims(spectrogram, axis=0)  # Add batch dimension

import numpy as np
from collections import Counter

# Predict using the model
predictions = new_model.predict(spectrogram)

# Get the predicted class indices
predicted_class_indices = np.argmax(predictions, axis=1)

# Count the occurrences of each class
class_counts = Counter(predicted_class_indices)

# Print the counts for each class
for class_index, count in class_counts.items():
    class_label = class_names[class_index]
    print(f"Class {class_label} ({class_index}): {count} times")


