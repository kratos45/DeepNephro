import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os

# Function to extract Mel Spectrogram
def extract_mel_spectrogram(file_path, save_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)  # Load audio
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Save the spectrogram as an image
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Extract spectrograms for all chunks
spectrogram_folder = "leakage_spectrograms"
os.makedirs(spectrogram_folder, exist_ok=True)

input_folder = "leakage_audio_chunks"  

for file in os.listdir(input_folder):
    if file.endswith(".wav"):
        file_path = os.path.join(input_folder, file)
        save_path = os.path.join(spectrogram_folder, f"{file.replace('.wav', '.png')}")
        
        print(f"Processing {file_path}...")  
        extract_mel_spectrogram(file_path, save_path)

print("Mel spectrograms generated!")
