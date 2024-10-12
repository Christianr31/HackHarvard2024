import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from pydub import AudioSegment
from collections import defaultdict
import numpy as np
import wave
import pylab
from pathlib import Path
from scipy import signal
from scipy.io import wavfile
from sklearn.metrics import confusion_matrix
import itertools
import shutil


metadata_path = 'model/data/Metadata_with_labels_SENT.csv'
sent_dir = 'model/data/SENT'

# Load the metadata
metadata = pd.read_csv(metadata_path)
# os.makedirs('spectrograms', exist_ok=True)
# remove all images in train and test dirs

# Remove all folders in the 'model/train' directory
train_dir = 'model/train'
for folder in os.listdir(train_dir):
    folder_path = os.path.join(train_dir, folder)
    if os.path.isdir(folder_path):
        shutil.rmtree(folder_path)

# Remove all folders in the 'model/test' directory
test_dir = 'model/test'
for folder in os.listdir(test_dir):
    folder_path = os.path.join(test_dir, folder)
    if os.path.isdir(folder_path):
        shutil.rmtree(folder_path)


os.makedirs('model/train/soft', exist_ok=True)
os.makedirs('model/train/normal', exist_ok=True)
os.makedirs('model/train/loud', exist_ok=True)
os.makedirs('model/train/veryloud', exist_ok=True)
os.makedirs('model/test/soft', exist_ok=True)
os.makedirs('model/test/normal', exist_ok=True)
os.makedirs('model/test/loud', exist_ok=True)
os.makedirs('model/test/veryloud', exist_ok=True)



# Print the columns to verify the column names
print(metadata.columns)



# Initialize counters and data storage
train_counts = defaultdict(int)
test_counts = defaultdict(int)
train_features = []
train_labels = []
test_features = []
test_labels = []

# Define the required number of samples
train_samples_per_category = 500
test_samples_per_category = 100
max_length = 200  # Define a fixed length for the spectrograms

train_counts = {label: 0 for label in metadata['Intensity_category_label'].unique()}
test_counts = {label: 0 for label in metadata['Intensity_category_label'].unique()}

# Function to check if all categories have reached their maximum counts
def all_categories_maxed(train_counts, test_counts, train_max, test_max):
    return all(count >= train_max for count in train_counts.values()) and \
           all(count >= test_max for count in test_counts.values())

def get_wav_info(wav_file):
    frame_rate, sound_info = wavfile.read(wav_file)
    return sound_info, frame_rate

def get_substring_after_last_underscore(s):
    """
    Returns the substring after the last instance of an underscore in the given string.
    
    Parameters:
    s (str): The input string.
    
    Returns:
    str: The substring after the last underscore, or the original string if no underscore is found.
    """
    if '_' in s:
        return s.rsplit('_', 1)[-1]
    return s

# count = 0
for index, row in metadata.iterrows():
    if all_categories_maxed(train_counts, test_counts, train_samples_per_category, test_samples_per_category):
        print("Collected required number of samples for all categories.")
        break
    # if count == 10:
    #     break
    file_name = row['Filename']
    label = row['Intensity_category_label']
    file_path = os.path.join(sent_dir, file_name)
    file_stem = Path(file_path).stem
    
    # Initialize a counter
file_counter = 0

for index, row in metadata.iterrows():
    if all_categories_maxed(train_counts, test_counts, train_samples_per_category, test_samples_per_category):
        print("Collected required number of samples for all categories.")
        break

    file_name = row['Filename']
    label = row['Intensity_category_label']
    file_path = os.path.join(sent_dir, file_name)
    file_stem = Path(file_path).stem

    if os.path.exists(file_path):
        if train_counts[label] < train_samples_per_category:
            # save spectrogram of this file
            sound_info, frame_rate = get_wav_info(file_path)
            Pxx, freqs, bins, im = plt.specgram(sound_info, Fs=frame_rate)
            plt.savefig(f'model/train/{label}/train_{file_stem}.png')
            train_counts[label] += 1
            plt.close()
            file_counter += 1
            if file_counter % 100 == 0:
                print(f"Processed {file_counter} files so far.")

        elif test_counts[label] < test_samples_per_category:
            sound_info, frame_rate = get_wav_info(file_path)
            Pxx, freqs, bins, im = plt.specgram(sound_info, Fs=frame_rate)
            plt.savefig(f'model/test/{label}/test_{file_stem}.png')
            test_counts[label] += 1
            plt.close()
            file_counter += 1
            if file_counter % 100 == 0:
                print(f"Processed {file_counter} files so far.")

        # Check if all categories have reached their maximum counts
        if all_categories_maxed(train_counts, test_counts, train_samples_per_category, test_samples_per_category):
            print("Collected required number of samples for all categories.")
            break
        # feature = extract_features(file_path)
        # if feature is not None:
        #     feature = pad_or_truncate_spectrogram(feature, max_length)
        #     if train_counts[label] < train_samples_per_category:
        #         train_features.append(feature)
        #         train_labels.append(label)
        #         train_counts[label] += 1
        #     elif test_counts[label] < test_samples_per_category:
        #         test_features.append(feature)
        #         test_labels.append(label)
        #         test_counts[label] += 1
            
        #     # print(f"Processed {file_name}")
            
        #     # Check if all categories have reached their maximum counts
        #     if all_categories_maxed(train_counts, test_counts, train_samples_per_category, test_samples_per_category):
        #         print("Collected required number of samples for all categories.")
        #         break

# # Debugging: Print the counts and shapes
# print(f"Train counts: {train_counts}")
# print(f"Test counts: {test_counts}")
# print(f"Train features shape: {len(train_features)}")
# print(f"Train labels shape: {len(train_labels)}")
# print(f"Test features shape: {len(test_features)}")
# print(f"Test labels shape: {len(test_labels)}")

# # Convert lists to tensors
# train_features = tf.stack(train_features)
# train_labels = tf.convert_to_tensor(train_labels)
# test_features = tf.stack(test_features)
# test_labels = tf.convert_to_tensor(test_labels)

# # Save the features and labels for training and testing
# np.save('model/data/train_features.npy', train_features.numpy())
# np.save('model/data/train_labels.npy', train_labels.numpy())
# np.save('model/data/test_features.npy', test_features.numpy())
# np.save('model/data/test_labels.npy', test_labels.numpy())

# # Plot the first 50 spectrograms from the training set
# if tf.size(train_features) > 0:
#     plot_spectrograms(train_features, title='First 50 Training Spectrograms', num_spectrograms=50)
# else:
#     print("No training features extracted.")

# def convert_to_pcm(file_path):
#     try:
#         audio = AudioSegment.from_file(file_path)
#         audio = audio.set_sample_width(2)  # Set sample width to 2 bytes (16 bits)
#         pcm_path = file_path.replace('.wav', '_pcm.wav')
#         audio.export(pcm_path, format='wav', codec='pcm_s16le')
#         return pcm_path
#     except Exception as e:
#         print(f"Error converting {file_path}: {e}")
#         return None

# def extract_features(file_path):
#     pcm_path = convert_to_pcm(file_path)
#     if pcm_path is None:
#         return None
#     audio_binary = tf.io.read_file(pcm_path)
#     audio, _ = tf.audio.decode_wav(audio_binary)
#     audio = tf.squeeze(audio, axis=-1)
#     stfts = tf.signal.stft(audio, frame_length=1024, frame_step=256, fft_length=1024)
#     spectrograms = tf.abs(stfts)
#     return spectrograms

# def pad_or_truncate_spectrogram(spectrogram, max_length):
#     current_length = tf.shape(spectrogram)[0]
#     if current_length > max_length:
#         return spectrogram[:max_length, :]
#     else:
#         padding = max_length - current_length
#         return tf.pad(spectrogram, [[0, padding], [0, 0]], mode='CONSTANT')

# def plot_spectrograms(spectrograms, title, num_spectrograms=50):
#     num_cols = 10
#     num_rows = (num_spectrograms + num_cols - 1) // num_cols  # Calculate the number of rows needed
#     fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 2 * num_rows))
#     axes = axes.flatten()
    
#     for i in range(num_spectrograms):
#         if i < len(spectrograms):
#             axes[i].imshow(tf.math.log(spectrograms[i] + 1e-6).numpy().T, aspect='auto', origin='lower')
#             axes[i].set_title(f'Spectrogram {i+1}')
#             axes[i].axis('off')
#         else:
#             axes[i].axis('off')
    
#     plt.suptitle(title)
#     plt.tight_layout()
#     plt.show()