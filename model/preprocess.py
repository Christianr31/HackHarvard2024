import os
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from pydub import AudioSegment

metadata_path = 'model/data/Metadata_with_labels_SENT.csv'
sent_dir = 'model/data/SENT'

# Load the metadata
metadata = pd.read_csv(metadata_path)

# Print the columns to verify the column names
print(metadata.columns)

def convert_to_pcm(file_path):
    try:
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_sample_width(2)  # Set sample width to 2 bytes (16 bits)
        pcm_path = file_path.replace('.wav', '_pcm.wav')
        audio.export(pcm_path, format='wav', codec='pcm_s16le')
        return pcm_path
    except Exception as e:
        print(f"Error converting {file_path}: {e}")
        return None

def extract_features(file_path):
    pcm_path = convert_to_pcm(file_path)
    if pcm_path is None:
        return None
    audio_binary = tf.io.read_file(pcm_path)
    audio, _ = tf.audio.decode_wav(audio_binary)
    audio = tf.squeeze(audio, axis=-1)
    stfts = tf.signal.stft(audio, frame_length=1024, frame_step=256, fft_length=1024)
    spectrograms = tf.abs(stfts)
    return spectrograms

def plot_spectrogram(spectrogram, title):
    plt.figure(figsize=(10, 6))
    plt.imshow(tf.math.log(spectrogram + 1e-6).numpy().T, aspect='auto', origin='lower')
    plt.title(title)
    plt.ylabel('Frequency bins')
    plt.xlabel('Time frames')
    plt.colorbar(format='%+2.0f dB')
    plt.show()

features = []
labels = []
for index, row in metadata.iterrows():
    file_name = row['Filename']
    label = row['Intensity_category_label']
    file_path = os.path.join(sent_dir, file_name)
    
    if os.path.exists(file_path):
        feature = extract_features(file_path)
        if feature is not None:
            features.append(feature)
            labels.append(label)
    print(f"Processed {file_name}")

# Convert lists to tensors
features = tf.stack(features)
labels = tf.convert_to_tensor(labels)

# Plot the first spectrogram
if features:
    plot_spectrogram(features[0], title='First Spectrogram')
else:
    print("No features extracted.")