import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

def download_audio(category):
    """
    Function to download the audiofiles in parallel from the preview urls in the stored JSON files
    """
    category=category.replace(" ","_")
    os.makedirs(f"../audiofiles/{category}",exist_ok=True) #if audiofiles folder doesn't already exist then create it
    filename=f"{category}_metadata.json"

    with open(f"../data/{filename}","r") as f:
        data=json.load(f)
        tasks=[]

        with ThreadPoolExecutor(max_workers=10) as executor: #execute downloads in parallel with 10 threads
            for i in range(len(data)):
                url=data[i]["previews"]["preview-lq-mp3"]
                filepath=f"../audiofiles/{category}/{category}_{i+1}.mp3"
                if os.path.exists(filepath): #if this file has already been downloaded, skip it
                    continue
                print(f"Downloading file {i+1} of {len(data)} with url: {url} from category: {category}")
                tasks.append(executor.submit(download_file_helper,url,filepath))

def download_file_helper(url,filepath):
    """
    Helper function to download a single audio file
    """
    try:
        urllib.request.urlretrieve(url, filepath)
    except Exception as e:
        print(f"Error! : {e}")


if __name__=="__main__":
    categories = [
        "rain sounds", "thunder sounds", "birds", "wind sounds", "dog bark", "cat meow",
        "car horn", "fireworks", "crowd noise", "applause", "laughter",
        "speech", "piano", "drums", "guitar", "footsteps", "sirens",
        "train sounds", "traffic sounds", "construction noise"
    ]

    for category in categories: 
        download_audio(category)

