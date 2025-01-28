import os
import librosa
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def preprocess_and_convert_audios(category,target_length=5,sample_rate=16000,trim_location="middle"):
    """
    Function to preprocess and convert all `.mp3` audio files in the specified category to `.wav` format.

    This function:
    - Reads all `.mp3` files in the input directory corresponding to the given `category`.
    - Trims or pads each audio file to a fixed length (`target_length` seconds, 5 seconds in our case).
    - Normalizes the audio amplitude for consistency.
    - Converts audio to `.wav` format with:
        - Mono channel
        - Specified sample rate (`sample_rate` Hz, 16000 Hz in our cases)
        - Default bit depth (16 bits).
    - Saves the preprocessed `.wav` files in the corresponding output directory.
    - Uses a thread pool with up to 16 workers to process multiple files in parallel for improved performance.

    Parameters:
    - category (str): The name of the category (e.g., "rain_sounds") whose files are to be processed.
    - target_length (int, optional): Target audio length in seconds (default is 5 seconds).
    - sample_rate (int, optional): Sampling rate for the output `.wav` files (default is 16,000 Hz).
    - trim_location (str, optional): Where to trim audio from ('start', 'middle', 'end'). Default is 'middle'.
    """
    
    input_dir=f"../audiofiles/{category}/"
    output_dir=f"../audiofiles_processed/{category}/"
    os.makedirs(output_dir,exist_ok=True)
    
    tasks=[]
    with ThreadPoolExecutor(max_workers=16) as executor: #execute the conversions in parallel with 16 threads, adjust number as needed (based on CPU)
        for file_name in os.listdir(input_dir):
            if file_name.endswith(".mp3"): #convert all mp3 files. avoid any other unnecessary or unsupported files 
                input_path = os.path.join(input_dir, file_name)
                output_path = os.path.join(output_dir, file_name.replace(".mp3", ".wav")) #output filepath name ends with .wav
                if os.path.exists(output_path): #if this file has already been preprocessed, skip it
                    continue
                #each executor calls preprocess_audio_helper
                tasks.append(executor.submit(preprocess_audio_helper,file_name,input_path,output_path, target_length, sample_rate, trim_location)) 
                
    
    for task in tasks: #wait for all threads to finish
        task.result() 

    print(f"All {category} audiofiles preprocessed successfully!")

def preprocess_audio_helper(file_name,input_path,output_path,target_length,sample_rate,trim_location):
    """
    Helper function to preprocess and save a single audio file.

    This function:
    - Loads the audio file from `input_path`.
    - Trims or pads the audio to the specified length (`target_length` seconds) based on the `trim_location` parameter.
    - Normalizes the amplitude of the audio.
    - Converts the audio to `.wav` format with the specified sample rate (`sample_rate` Hz).
    - Saves the preprocessed `.wav` file to `output_path`.

    Parameters:
    - file_name (str): Name of the audio file being processed (for logging).
    - input_path (str): Path to the source `.mp3` file.
    - output_path (str): Path to save the preprocessed `.wav` file.
    - target_length (int): Target audio length in seconds.
    - sample_rate (int): Sampling rate for the `.wav` file.
    - trim_location (str): Where to trim audio from ('start', 'middle', 'end').
    """

    try:
        audio,sr=librosa.load(input_path,sr=sample_rate,mono=True) #load the MP3 file and convert it to a mono audio signal with the specified sample rate
        
        max_length=target_length*sample_rate #calculate the target number of samples based on the target length and sample rate
        audio=trim_or_pad_audio(audio,max_length,trim_location) #trim or pad the audio to max_length

        audio=librosa.util.normalize(audio) #normalize the amplitude

        sf.write(output_path, audio, samplerate=sample_rate) #save as wav file
        print(f"Processed and Converted: {file_name} -> {output_path}")
    
    except Exception as e:
        print(f"Error processing {file_name}: {e}")

def trim_or_pad_audio(audio, max_length, trim_location):
    """
    Trims or pads an audio signal to a fixed length.

    This function:
    - Trims the audio from the beginning, middle, or end if it exceeds the target length.
    - Pads the audio with zeros equally on both sides if it is shorter than the target length.

    Parameters:
    - audio (numpy.ndarray): The audio signal as a NumPy array.
    - max_length (int): The target number of samples (calculated as `target_length * sample_rate`).
    - trim_location (str): Where to trim the audio from ('start', 'middle', 'end').

    Returns:
    - numpy.ndarray: The trimmed or padded audio signal.
    """
    
    if len(audio)>max_length: #if length exceeds max_length then trim the audio 
        if trim_location=="start": #we take the specified length from the beginning (5 seconds in our case)
            audio=audio[:max_length] 
        elif trim_location=="middle": #we take the specified length from the middle (5 seconds in our case)
            mid=len(audio)//2
            samples_per_side=max_length//2
            start=mid-samples_per_side
            end=mid+samples_per_side
            audio=audio[start:end]
        elif trim_location=="end": #we take the specified length from the end (5 seconds in our case)
            audio=audio[-max_length:]
        else:
            raise Exception("Invalid trim location")
    else: #if length is shorter than max_length then we pad the audio from both the beginning and the end
        padding=max_length-len(audio)
        pad_left=padding//2
        pad_right=padding-pad_left
        audio=np.pad(audio, (pad_left, pad_right), mode='constant')

    return audio


if __name__=="__main__":
    with open("categories.txt","r") as f:
        categories=f.read().split()
    
    for category in categories:
        preprocess_and_convert_audios(category)