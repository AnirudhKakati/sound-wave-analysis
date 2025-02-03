import os
import pandas as pd
import numpy as np
import librosa
from concurrent.futures import ThreadPoolExecutor

def get_audio_features(category):
    """
    Function to extract features from all audio files in a given category and save to CSV.
    Features extracted:
    - MFCCs (Mel-Frequency Cepstral Coefficients)
    - Chroma Features
    - Spectral Contrast
    - Zero-Crossing Rate
    - Spectral Centroid
    - Spectral Bandwidth
    - RMS Energy

    Parameters:
    - category (str): The category folder containing `.wav` files.
    """

    #output path where we will save the CSV
    output_dir="../audiofiles_processed_features_CSVs"
    os.makedirs(output_dir,exist_ok=True)
    output_csv=f"{output_dir}/{category}_features.csv"

    if os.path.exists(output_csv): #if the csv already exists then we return from the function
        print(f"Feature extraction skipped for category: {category}. CSV already exists: {output_csv}")
        return

    base_path=f"../audiofiles_processed/{category}/"
    print(f"Extracting features for category: {category}")
    
    data=[]
    with ThreadPoolExecutor(max_workers=16) as executor: #execute feature extraction in parallel with 16 threads, adjust number as needed (based on CPU)
        tasks={}
        for filename in os.listdir(base_path):
            if filename.endswith(".wav"):
                filepath=os.path.join(base_path, filename) 
                tasks[filename]=executor.submit(get_audio_features_helper,filepath)

        for filepath,task in tasks.items():
            features=task.result()
            if features is not None:
                data.append([filepath]+list(features))
        
    #we define column names
    columns=["filename"]+[f"mfcc_{i+1}" for i in range(13)]+[f"chroma_{i+1}" for i in range(12)]+[f"spectral_contrast_{i+1}" for i in range(7)]+["zero_crossing_rate", "spectral_centroid", "spectral_bandwidth", "rms_energy"]

    # save to csv files
    df=pd.DataFrame(data, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"Feature extraction complete for category: {category}. Saved to {output_csv}")


def get_audio_features_helper(filepath,sample_rate=16000):
    """
    Helper function to extracts audio features from a given `.wav` file.

    Parameters:
    - audio_path (str): Path to the `.wav` file.
    - sample_rate (int, optional): Sampling rate for loading the audio (default is 16,000 Hz).
    
    Returns:
    - list: A list containing extracted features for the audio file.
    """

    try:
        print(f"Extracting features from {filepath}")
        y,sr=librosa.load(filepath,sr=sample_rate,mono=True)

        mfccs=np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
        chroma=np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        spectral_contrast=np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
        zero_crossing_rate=np.mean(librosa.feature.zero_crossing_rate(y=y))
        spectral_centroid=np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_bandwidth=np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        rms_energy=np.mean(librosa.feature.rms(y=y))
        
        #concatenate all features into a single vector
        features=np.hstack([mfccs, chroma, spectral_contrast, zero_crossing_rate, spectral_centroid, spectral_bandwidth, rms_energy])    
        return features
    
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

if __name__=="__main__":
    with open("categories.txt","r") as f:
        categories=f.read().split()
    
    for category in categories:
        get_audio_features(category)