import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

def convert_audios(category):
    """
    Function to convert all `.mp3` audio files in the specified category to `.wav` format with 
    - Sample rate set to 16,000 Hz
    - Mono audio channel

    This function:
    - Reads all `.mp3` files in the input directory corresponding to the given `category`.
    - Converts each file to `.wav` format and saves it in the corresponding output directory.
    - Skips files that have already been converted.
    - Uses a thread pool with up to 16 workers to process multiple files in parallel for improved performance.
    - Waits for all threads to finish before printing the success message.

    Parameters:
    - category (str): The name of the category (e.g., "rain_sounds") whose files are to be converted.
    """
    
    input_dir=f"../audiofiles/{category}/"
    output_dir=f"../audiofiles_converted/{category}/"
    os.makedirs(output_dir,exist_ok=True)
    
    tasks=[]
    with ThreadPoolExecutor(max_workers=16) as executor: #execute the conversions in parallel with 16 threads
        for file_name in os.listdir(input_dir):
            if file_name.endswith(".mp3"): #convert all mp3 files. avoid any other unnecessary files that may have been accidentally added 
                input_path = os.path.join(input_dir, file_name)
                output_path = os.path.join(output_dir, file_name.replace(".mp3", ".wav")) #output filepath name ends with .wav
                if os.path.exists(output_path): #if this file has already been converted, skip it
                    continue
                tasks.append(executor.submit(convert_audio_helper,file_name,input_path,output_path)) #each executor calls convert_audio_helper
    
    for task in tasks: #wait for all threads to finish
        task.result() 

    print(f"All {category} audiofiles converted successfully!")

def convert_audio_helper(file_name,input_path,output_path):
    """
    Helper function to convert a single audio file using FFmpeg.

    This function:
    - Converts the audio file at `input_path` to the `.wav` format at `output_path`.
    - Sets the sample rate to 16,000 Hz (`-ar 16000`) for consistent audio quality.
    - Forces the audio channel to mono (`-ac 1`) to simplify the processing of multi-channel audio.
    - Uses `subprocess.run` to execute the FFmpeg command with `check=True`:
    - Ensures that any failure in FFmpeg raises a `CalledProcessError`.
    - Captures the output and errors for debugging with `stdout=subprocess.PIPE` and `stderr=subprocess.PIPE` 

    Parameters:
    - file_name (str): Name of the audio file being processed (for logging).
    - input_path (str): Path to the source `.mp3` file.
    - output_path (str): Path where the converted `.wav` file will be saved.   
    """
    
    try:
        subprocess.run( 
            ["ffmpeg", "-i", input_path, "-ar", "16000", "-ac", "1", output_path], #converts to .wav format, sets sample rate to 16000Hz and audio channel to mono
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        print(f"Converted: {file_name} -> {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting {file_name}: {e.stderr.decode()}")

if __name__=="__main__":
    with open("categories.txt","r") as f:
        categories=f.read().split()
    
    for category in categories:
        convert_audios(category)