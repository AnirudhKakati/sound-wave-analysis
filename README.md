# Sound Wave Analysis

Checkout the project website:  
👉 [https://anirudhkakati.github.io/sound-wave-analysis/index.html](https://anirudhkakati.github.io/sound-wave-analysis/index.html)

This project explores the fascinating world of sound waves by analyzing and classifying various sound categories using machine learning techniques. It covers the full journey from raw data collection to building a public-facing website.

---

## Requirements

- **Python 3.11.6**  
  It is recommended to use Python 3.10 or 3.11, as some libraries used may not fully support Python versions 3.12 and above.  
  [Download Python 3.11.6](https://www.python.org/downloads/release/python-3116/)

- **FFmpeg**  
  Required for audio conversion.  
  [Download FFmpeg](https://ffmpeg.org/download.html)

- **Install all required libraries:**
  ```bash
  pip install -r requirements.txt
  ```

- **Additional:**  
  A `.env` file is required containing the Freesound API key.

---

## Project Structure

```plaintext
project-root/
|-- audiofiles/                             # (External) Raw audio files (.mp3) (Google Drive)
|-- audiofiles_processed/                   # (External) Preprocessed audio files (.wav) (Google Drive)
|-- audiofiles_metadata/
|   |-- json_files/                         # Metadata JSON files
|   |-- csv_converted_files/                 # Metadata CSV files
|-- audiofiles_processed_features_CSVs/      # Extracted audio feature CSVs
|-- audio_features_transaction_form/         # Transactions for ARM
|-- scripts/                                 # All project scripts
|   |-- getting_metadata.py
|   |-- getting_audiofiles.py
|   |-- preprocessing_audiofiles.py
|   |-- extracting_audio_features.py
|   |-- visualizations.py
|   |-- pca.py
|   |-- clustering.py
|   |-- arm.py
|   |-- naive_bayes.py
|   |-- decision_tree.py
|   |-- regression.py
|   |-- svm.py
|   |-- ensemble.py
|-- website/                                # Frontend website
|   |-- plots/
|   |-- data/
|   |-- css/
|   |-- js/
|   |-- images/
|   |-- index.html and other tabs
|-- README.md                               # Project documentation
|-- requirements.txt                        # Python dependencies
|-- categories.txt                          # List of sound categories
|-- .env                                    # (local) Freesound API key
```

> **Note:** `audiofiles/` and `audiofiles_processed/` folders are not part of the GitHub repository. They are available via external Google Drive links.

---

## How to Run

Before starting, ensure:
- Python 3.10/3.11 is installed.
- FFmpeg is installed and working.
- Dependencies are installed via `pip install -r requirements.txt`.
- Freesound API key is set up in `.env` file.


### Step 1: Metadata Collection
Fetch metadata for different sound categories from Freesound API.
```bash
python scripts/getting_metadata.py
```

### Step 2: Audio File Download
Download audio files based on collected metadata.
```bash
python scripts/getting_audiofiles.py
```

### Step 3: Audio Preprocessing
Convert `.mp3` files to `.wav`, standardize sampling rate, length, and normalize.
```bash
python scripts/preprocessing_audiofiles.py
```

### Step 4: Feature Extraction
Extract features like MFCCs, Chroma, Spectral features, etc.
```bash
python scripts/extracting_audio_features.py
```

### Step 5: Data Visualization
Generate plots like MFCC radial plots, spectral features, RMS energy graphs.
```bash
python scripts/visualizations.py
```

---

## Scripts Description

### Initial Scripts
- `getting_metadata.py` → Fetches metadata for specified sound categories.
- `getting_audiofiles.py` → Downloads audio files using metadata.
- `preprocessing_audiofiles.py` → Converts and preprocesses audio files.
- `extracting_audio_features.py` → Extracts structured audio features into CSVs.
- `visualizations.py` → Generates various plots based on extracted features.

### Modeling Scripts
- `pca.py` → Performs Principal Component Analysis (PCA) for dimensionality reduction.
- `clustering.py` → Applies KMeans, Hierarchical, and DBSCAN clustering.
- `arm.py` → Performs Association Rule Mining (ARM) on extracted features.
- `naive_bayes.py` → Trains and evaluates Multinomial, Gaussian, and Bernoulli Naive Bayes models.
- `decision_tree.py` → Trains Decision Tree classifiers with different hyperparameters.
- `regression.py` → Compares Logistic Regression and Multinomial Naive Bayes models.
- `svm.py` → Trains Support Vector Machine models with linear, RBF, and polynomial kernels.
- `ensemble.py` → Trains a Random Forest ensemble classifier.

> Each script can be run independently based on the desired analysis or model building.

---

## Website Navigation

The interactive website provides an engaging way to explore the project:

- **Introduction** → Overview of the project goals and motivations.
- **Data Preparation & Exploration** → Steps taken to collect, process, and visualize data.
- **Unsupervised Learning** → PCA visualizations, clustering results, association rule mining.
- **Supervised Learning** → Classification results using Naive Bayes, Decision Trees, Regression, SVM, and Random Forest.
- **Conclusion** → Key takeaways and final thoughts.
- **About Me** → Background of the creator.

---

## External Links

- **Google Drive Links:**
  - [Raw Audio Files (.mp3)](https://drive.google.com/drive/folders/1Nw9VAKk4MGyr95R4O1fodtEa8DFvYOl8)
  - [Processed Audio Files (.wav)](https://drive.google.com/drive/folders/1M4nXU0G0at0M0Lx15LT-ubIfEF-DtEmD)

- **Project Website:**  
  [https://anirudhkakati.github.io/sound-wave-analysis/index.html](https://anirudhkakati.github.io/sound-wave-analysis/index.html)

---

#
