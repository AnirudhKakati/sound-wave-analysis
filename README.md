# Sound Wave Analysis (WORK IN PROGRESS)

This project involves analyzing and classifying sound waves from various categories using machine learning techniques. The initial steps include data collection, preprocessing, and conversion to prepare the dataset for further analysis.

## Requirements
- Python 3.11.6 ([Download Python 3.11.6](https://www.python.org/downloads/release/python-3116/))
  - It is recommended to use Python 3.10 or 3.11 as some libraries used in the project may not yet fully support Python versions 3.12 and above.
- FFmpeg ([Download FFmpeg](https://ffmpeg.org/download.html))
- Required Python libraries:
  ```bash
  pip install -r requirements.txt
  ```

## Steps Completed So Far

### 1. Metadata Collection
Metadata for audio files was collected using the Freesound API. The metadata includes categories and relevant details, saved as JSON files for further processing.

### 2. Audio File Download
Audio files in `.mp3` format were downloaded from Freesound based on the metadata.

### 3. Audio File Conversion and Preprocessing
Downloaded `.mp3` files were converted to `.wav` format and preprocessed to standardize the dataset. Preprocessing steps included:
- Converting to mono audio with a sample rate of 16,000 Hz.
- Standardizing audio length to 5 seconds by trimming or padding.
- Normalizing the amplitude for consistent volume levels.

The preprocessed files are stored as `.wav` files in category-specific directories.

## Directory Structure
```plaintext
project-root/
|-- data/                        # Metadata JSON files for each category
|-- scripts/                     # Scripts for data collection, preprocessing, and conversion
|-- README.md                    # Project documentation
```

The audio files and preprocessed audio files are stored externally due to size constraints and can be accessed via Google Drive:
- [Raw Audio Files (.mp3)](https://drive.google.com/drive/folders/1Nw9VAKk4MGyr95R4O1fodtEa8DFvYOl8)
- [Processed Audio Files (.wav)](<add_google_drive_link_here>)

## How to Run
- All dependencies are included in the requirements.txt file. Before running the scripts create a virtual environment (with Python 3.10 or 3.11) and run
pip install -r requirements.txt to ensure all necessary packages are installed. Also ensure ffmpeg is installed and working.

1. **Metadata Collection**:
   Run the metadata collection script to fetch metadata for the specified categories:
   ```bash
   python scripts/getting_metadata.py
   ```
2. **Download Audio Files**:
   Download audio files using the metadata:
   ```bash
   python scripts/getting_audiofiles.py
   ```
3. **Preprocess and Convert Audio Files**:
   Convert and preprocess audio files:
   ```bash
   python scripts/preprocessing_audiofiles.py
   ```

---
