import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

def download_audio(category):
    """
    Function to download all audio files for the specified category in parallel using URLs stored in the metadata JSON file.

    This function:
    - Reads the metadata JSON file for the given `category`.
    - Extracts the preview URLs for audio files.
    - Downloads the audio files in parallel using a thread pool with up to 16 workers.
    - Skips files that have already been downloaded.

    Parameters:
    - category (str): The name of the category (e.g., "rain_sounds") whose audio files are to be downloaded.
    """

    os.makedirs(f"../audiofiles/{category}",exist_ok=True) #if audiofiles folder doesn't already exist then create it
    filename=f"{category}_metadata.json"

    with open(f"../audiofiles_metadata/json_files/{filename}","r") as f:
        data=json.load(f)
        tasks=[]

        with ThreadPoolExecutor(max_workers=16) as executor: #execute downloads in parallel with 16 threads, adjust number as needed (based on CPU)
            for i in range(len(data)):
                url=data[i]["previews"]["preview-lq-mp3"]
                filepath=f"../audiofiles/{category}/{category}_{i+1}.mp3"
                if os.path.exists(filepath): #if this file has already been downloaded, skip it
                    continue
                print(f"Downloading file {i+1} of {len(data)} with url: {url} from category: {category}")
                tasks.append(executor.submit(download_file_helper,url,filepath)) #each executor calls download_file_helper
    print(f"Successfully downloaded all audiofiles of category: {category}")

def download_file_helper(url,filepath):
    """
    Helper function to download a single audio file from the given URL and saves it to the specified file path.

    This function:
    - Retrieves the file from the `url` using `urllib.request.urlretrieve`.
    - Saves the file to the location specified by `filepath`.

    Parameters:
    - url (str): The URL of the audio file to be downloaded.
    - filepath (str): The path where the downloaded file should be saved.
    """

    try:
        urllib.request.urlretrieve(url, filepath)
    except Exception as e:
        print(f"Error! : {e}")


if __name__=="__main__":
    with open("categories.txt","r") as f: #the categories are mentioned in categories.txt file
        categories = f.read().split()

    for category in categories: 
        download_audio(category)

