import requests
import json
import os

# the next 3 lines can be skipped by replacing API_key value with the generated API key from Freesound API
# here the API key was stored in a .env file and retrieved
from dotenv import load_dotenv
load_dotenv() 
API_KEY=os.getenv("API_KEY") #replace with the generated API key

BASE_URL="https://freesound.org/apiv2"

def search_sounds(query,max_results=150, page=1):
    """
    Function to search and fetch with pagination the metadata of specific sounds from Freesound API
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
    Function to save the fetched metadata of sounds into a JSON file
    """

    os.makedirs("../data",exist_ok=True) #if data folder doesn't already exist then create it
    filename=f"../data/{category.replace(" ","_")}_metadata.json"
    with open(filename,"w") as f:
        json.dump(data,f,indent=4)
    print(f"Metadata saved for category: {category}")

def get_category_data(category,total_results=750):
    """
    Function to get and save the metadata of 750 sounds of a specific category from Freesound API. 
    Calls search_sounds function to fetch the sounds and save_metadata function to save them.
    """

    all_data=[]
    page=1 #to keep track of the current page to fetch
    while len(all_data)<total_results: #keep fetching till we have atleast 750 sounds
        data,next_page=search_sounds(category,page=page)
        all_data.extend(data)
        if not next_page: #if no more pages to fetch then break
            break
        page+=1 #move to the next page to fetch
    
    if len(all_data)<total_results: # if the loop breaks early (i.e. not enough data) then log that
        print("Not enough data to fetch!")

    save_metadata(all_data[:total_results],category) #save the first 750 results

if __name__=="__main__":
    with open("categories.txt","r") as f: #the categories are mentioned in categories.txt file
        categories = f.read().split()
    
    for category in categories:
        print(f"Fetching data for category: {category}")
        get_category_data(category)