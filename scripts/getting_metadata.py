import requests
import json
import os

# The next 3 lines can be skipped by replacing API_key value with the generated API key from Freesound API
# Here the API key was stored in a .env file and retrieved. To follow this structure, create a .env file in the main project folder with
# the contents API_KEY = <YOUR-API-KEY>. Replace "<YOUR-API-KEY>" with the actual generated API key
from dotenv import load_dotenv
load_dotenv() 
API_KEY=os.getenv("API_KEY") # or replace this with the generated API key and skip this process

BASE_URL="https://freesound.org/apiv2"

def search_sounds(query,max_results=150, page=1):
    """
    Function to search for sounds using the Freesound API and retrieves metadata for a specific query.

    This function:
    - Sends a GET request to the Freesound API's search endpoint.
    - Retrieves metadata for sounds based on the provided `query`.
    - Handles pagination to fetch results for the specified page.
    - Returns the results and the link to the next page (if available).

    Parameters:
    - query (str): The search term to query the Freesound API.
    - max_results (int, optional): Maximum number of results per page (default is 150, the API's maximum limit).
    - page (int, optional): The page number to fetch (default is 1).
    
    Returns:
    - tuple: A tuple containing:
        - results (list): A list of sound metadata dictionaries.
        - next_page (str or None): The URL for the next page of results, or None if there are no more pages.
    """

    url=f"{BASE_URL}/search/text/"
    params={
        "query":query,
        "fields": "id,name,tags,description,duration,previews,license,url",
        "page_size": max_results, #150 is the maximum for each page
        "page": page, #which page to fetch will be passed each time this is called
        "token": API_KEY
    }

    response=requests.get(url,params)

    if response.status_code==200:
        data=response.json()
        return data.get("results",[]),data.get("next",None) #return the data and the next page to fetch
    else:
        print(f"Error! Status Code: {response.status_code} - {response.text}")
        return [],None

def save_metadata(data,category):
    """
    Function to saves the metadata of sounds into a JSON file for the specified category.

    This function:
    - Creates a `../data` directory if it doesn’t already exist.
    - Writes the provided metadata to a JSON file named `{category}_metadata.json`.

    Parameters:
    - data (list): A list of sound metadata dictionaries to save.
    - category (str): The name of the category (e.g., "rain_sounds") to which the metadata belongs.
    """

    os.makedirs("../data",exist_ok=True) #if data folder doesn't already exist then create it
    filename=f"../data/{category}_metadata.json"
    with open(filename,"w") as f:
        json.dump(data,f,indent=4)
    print(f"Metadata saved for category: {category}")

def get_category_data(category,total_results=750):
    """
    Function to fetch and save metadata for a specified number of sounds in a given category from the Freesound API.

    This function:
    - Calls `search_sounds` to fetch sound metadata in batches (paginated).
    - Aggregates the metadata until the desired number of results (`total_results`) is collected.
    - Calls `save_metadata` to save the metadata to a JSON file for the specified category.

    Parameters:
    - category (str): The name of the category (e.g., "rain_sounds") whose data is to be fetched.
    - total_results (int, optional): The total number of results to fetch (default is 750).
    """

    all_data=[]
    page=1 #to keep track of the current page to fetch
    while len(all_data)<total_results: #keep fetching till we have atleast total_results (750 in our case) sounds 
        data,next_page=search_sounds(category,page=page)
        all_data.extend(data)
        if not next_page: #if no more pages to fetch then break
            break
        page+=1 #move to the next page to fetch
    
    if len(all_data)<total_results: # if the loop breaks early (i.e. not enough data) then log that
        print("Not enough data to fetch!")

    save_metadata(all_data[:total_results],category) #save the first total_results (750 in our case) results

if __name__=="__main__":
    with open("categories.txt","r") as f: #the categories are mentioned in categories.txt file
        categories = f.read().split()
    
    for category in categories:
        print(f"Fetching data for category: {category}")
        get_category_data(category)