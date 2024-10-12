import numpy as np
import tensorflow as tf

from pydub import AudioSegment


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

def pad_or_truncate_spectrogram(spectrogram, max_length):
    current_length = tf.shape(spectrogram)[0]
    if current_length > max_length:
        return spectrogram[:max_length, :]
    else:
        padding = max_length - current_length
        return tf.pad(spectrogram, [[0, padding], [0, 0]], mode='CONSTANT')

# Reload a fresh Keras model from the .keras zip archive
new_model = tf.keras.models.load_model('my_model.keras')

# Show the model architecture
new_model.summary()

max_length = 200
# Audio file to test
file_path = 'test.wav'
feature = extract_features(file_path)
if feature is not None:
    feature = pad_or_truncate_spectrogram(feature, max_length)
else:
    print("Error extracting features from the audio file")
    exit()

# Normalize the feature
feature = feature / np.max(feature)

# Expand dimensions to match the input shape of the pretrained model
feature = np.expand_dims(feature, axis=-1)

# Convert single-channel spectrograms to 3-channel by repeating the single channel
feature = np.repeat(feature, 3, axis=-1)

# Make a prediction
predictions = new_model.predict(np.array([feature]))
predicted_label = np.argmax(predictions)

print(f'Predicted label: {predicted_label}')