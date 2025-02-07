import os
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg before importing pyplot
# Set the backend to 'Agg' which is:
# 1. Non-interactive: doesn't try to open windows or create displays
# 2. Thread-safe: works safely with multi-threaded applications
# 3. Memory-efficient: doesn't keep references to previous plots
# 4. Written in C++: faster than other backends
# since we are generating plots and saving them, and not really displaying interactive plots, this backend is ideal 
import matplotlib.pyplot as plt

import seaborn as sns
import librosa
import librosa.display
from concurrent.futures import ThreadPoolExecutor, as_completed

feature_dir="../audiofiles_processed_features_CSVs/" #path to processed feature CSVs
base_dir="../audiofiles_processed/" #path to the audiofiles
with open("../scripts/categories.txt","r") as f: # get the categories
    categories=f.read().split() 


def load_features(category):
    """
    Function to load extracted audio features for a given category from a CSV file.

    This function:
    - Constructs the file path for the category's feature CSV.
    - Reads the CSV file into a pandas DataFrame.
    - Returns the DataFrame containing extracted audio features.

    Parameters:
    - category (str): The category name (e.g., "rain_sounds", "wind_sounds").

    Returns:
    - df (pd.DataFrame): A DataFrame containing the features for the given category.
    """

    file_path=os.path.join(feature_dir, f"{category}_features.csv")
    df=pd.read_csv(file_path)
    return df


def load_combined_features():
    """
    Function to load and combine extracted audio features for all categories.

    This function:
    - Iterates over all predefined categories.
    - Loads the feature CSV for each category.
    - Appends a "Category" column to distinguish data.
    - Combines all category DataFrames into a single DataFrame.

    Returns:
    - pd.DataFrame: A DataFrame containing combined features from all categories.
    """

    dataframes=[]
    
    for category in categories:
        file_path=os.path.join(feature_dir, f"{category}_features.csv")
        df=pd.read_csv(file_path)

        df["Category"]=category #add a column for category
        dataframes.append(df)

    return pd.concat(dataframes, ignore_index=True)


def plot_mfcc_radial(category):
    """
    Function to generate and save a radial plot of the Mel-Frequency Cepstral Coefficients (MFCCs) for a given category.

    This function:
    - Loads MFCC feature data for the specified category.
    - Computes the mean MFCC values across all audio samples.
    - Creates a radial (spider) plot to visualize MFCC distribution.
    - Enhances visualization with dynamic text labels, colors, and styling.
    - Saves the generated plot as a PNG file.

    Parameters:
    - category (str): The category name (e.g., "rain_sounds", "wind_sounds") for which the MFCC plot is generated.

    Saves:
    - A radial MFCC plot as a PNG file in `../website/plots/mfcc_radial/`.
    """

    df=load_features(category)

    #extract MFCC features (mean values across all audio samples)
    mfcc_cols=[col for col in df.columns if "mfcc" in col]
    mfcc_means=df[mfcc_cols].mean().values

    #define angles for the radial plot
    angles=np.linspace(0, 2 * np.pi, len(mfcc_means), endpoint=False).tolist()
    angles += angles[:1]  # Close the circle

    #repeat first value to close the circular shape
    mfcc_means=np.append(mfcc_means, mfcc_means[0])


    fig, ax=plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))    
    cmap=plt.get_cmap("viridis")
    color=cmap(0.6)
    
    ax.fill(angles, mfcc_means, color=color, alpha=0.5)  # Soft fill
    ax.plot(angles, mfcc_means, color=color, linewidth=1, linestyle="-")  # Bolder outline
    
    # Fix tick label alignment
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])
    
    # Dynamically adjust label angles
    # Adjust label angles to align along the tangent at each point
    for angle, label in zip(angles[:-1], mfcc_cols):
        rotation=np.degrees(angle)+90  # Rotate perpendicular to radius (tangent)
        if 90 <= rotation <= 270:  # bottom-side labels (flip)
            rotation += 180

        ax.text(angle, 200 * 1.1, label, 
                ha="center", va="center", 
                fontsize=10, fontweight="bold", color="black", rotation=rotation)
        
    # add numerical values near each point
    for angle, value in zip(angles[:-1], mfcc_means[:-1]):  # skip last duplicate value
        ax.text(angle, value * 1.05, f"{value:.2f}", 
                ha="center", va="center", fontsize=7, fontweight="bold", 
                color="black")

    ax.set_yticklabels([])  #hide radial grid labels

    ax.set_ylim(-400, 200)
    ax.set_title(f"MFCC Radial Plot - {category}", fontsize=14, fontweight="bold", pad=20, color=color)

    #save the enhanced plot
    output_path=f"../website/plots/mfcc_radial"
    os.makedirs(output_path,exist_ok=True)
    plot_filename=f"mfcc_radial_{category}.png"
    plt.savefig(f"{output_path}/{plot_filename}", dpi=300, bbox_inches="tight")
    print(f"Saved MFCC radial plot: {plot_filename}")

    plt.close()


def plot_3d_spectral_features():
    """
    Function to generate and save a 3D scatter plot of spectral features for all categories.

    This function:
    - Loads spectral feature data for each category.
    - Computes the mean values for Spectral Centroid, Spectral Bandwidth, and RMS Energy.
    - Creates a 3D scatter plot to visualize the distribution of these features.
    - Uses distinct colors for each category and enhances visibility with labels and edge styling.
    - Saves the generated plot as a PNG file.

    Saves:
    - A 3D spectral feature plot as `spectral_3d_plot.png` in `../website/plots/spectral_3d_plot/`.
    """

    fig=plt.figure(figsize=(10, 7))
    ax=fig.add_subplot(111, projection="3d")

    colors=plt.cm.tab20(np.linspace(0, 1, len(categories)))  #uses the `tab20` colormap for distinct category colors

    for category, color in zip(categories, colors):
        df=load_features(category)

        #extract mean values for spectral features
        spectral_centroid=df["spectral_centroid"].mean()
        spectral_bandwidth=df["spectral_bandwidth"].mean()
        rms_energy=df["rms_energy"].mean()

        ax.scatter(spectral_centroid, spectral_bandwidth, rms_energy, 
                   color=color, label=category, s=100, edgecolor="black", alpha=0.8)

    ax.set_xlabel("Spectral Centroid", fontsize=12, fontweight="bold")
    ax.set_ylabel("Spectral Bandwidth", fontsize=12, fontweight="bold")
    ax.set_zlabel("RMS Energy", fontsize=12, fontweight="bold")
    ax.set_title("3D Spectral Feature Plot", fontsize=14, fontweight="bold", pad=20)

    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1))

    #save the plot
    output_path=f"../website/plots/spectral_3d_plot"
    os.makedirs(output_path,exist_ok=True)
    plot_filename="spectral_3d_plot.png"
    plt.savefig(f"{output_path}/{plot_filename}", dpi=300, bbox_inches="tight")
    print(f"Saved 3D Spectral Feature Plot: {plot_filename}")

    plt.close()


def plot_chroma_heatmap():
    """
    Function to generate and save a heatmap of chroma feature means for all categories.

    This function:
    - Loads chroma feature data for each category.
    - Computes the mean values for all chroma features across audio samples.
    - Constructs a DataFrame for visualization.
    - Plots a heatmap using the "coolwarm" colormap with annotations.
    - Saves the generated plot as a PNG file.

    Saves:
    - A chroma feature heatmap as `chroma_heatmap.png` in `../website/plots/chroma_heatmap/`.
    """

    chroma_means=[]

    for category in categories:
        df=load_features(category)

        #extract mean chroma values for this category
        chroma_cols=[col for col in df.columns if "chroma" in col]
        chroma_values=df[chroma_cols].mean().values

        chroma_means.append(chroma_values)

    #convert to DataFrame for heatmap
    chroma_df=pd.DataFrame(chroma_means, index=categories, columns=chroma_cols)

    #plot heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(chroma_df, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5)

    plt.title("Chroma Feature Heatmap", fontsize=14, fontweight="bold")
    plt.xlabel("Chroma Features", fontsize=12, fontweight="bold")
    plt.ylabel("Sound Categories", fontsize=12, fontweight="bold")

    #save the plot
    plot_filename="chroma_heatmap.png"
    output_path=f"../website/plots/chroma_heatmap"
    os.makedirs(output_path,exist_ok=True)
    plt.savefig(f"{output_path}/{plot_filename}", dpi=300, bbox_inches="tight")
    print(f"Saved Chroma Feature Heatmap: {plot_filename}")

    plt.close()


def compute_spectrogram(file_path):
    """
    Function to compute the magnitude spectrogram of an audio file.

    This function:
    - Loads an audio file using Librosa.
    - Computes the Short-Time Fourier Transform (STFT) to obtain the magnitude spectrogram.

    Parameters:
    - file_path (str): Path to the `.wav` audio file.

    Returns:
    - tuple:
        - np.ndarray: The computed magnitude spectrogram.
        - int: The sample rate of the audio file.
    """

    try:
        y,sr=librosa.load(file_path, sr=None) #load the audio file (uses the original sample rate (`sr=None`) when loading audio)
        D=np.abs(librosa.stft(y))  #get magnitude spectrogram
        return D, sr
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None
    
def plot_avg_spectrogram(category):
    """
    Function to generate and save an average spectrogram plot for a given category

    This function:
    - Iterates over all `.wav` files in the specified category.
    - Computes the spectrogram for each file using parallel processing.
    - Averages the spectrograms across all files.
    - Converts the averaged spectrogram to a dB scale.
    - Plots the spectrogram using a logarithmic frequency scale.
    - Saves the generated plot as a PNG file.

    Parameters:
    - category (str): The category name (e.g., "rain_sounds", "wind_sounds").

    Saves:
    - An average spectrogram plot as `spectrogram_avg_{category}.png` in `../website/plots/average_spectogram/`.
    """

    category_path=os.path.join(base_dir, category)
    files=[f for f in os.listdir(category_path) if f.endswith(".wav")]

    if not files:
        print(f"No audio files found for category: {category}")
        return

    # initialize empty list to store spectrograms and variable for sample rate
    spectrograms=[]
    sr=None

    # create a thread pool for parallel computation
    # max_workers=8 limits the number of concurrent threads, adjust number based on CPU
    with ThreadPoolExecutor(max_workers=8) as executor:
        # create a dictionary mapping Future objects to their corresponding files
        # this allows us to track which file each result came from
        future_to_file={
            executor.submit(compute_spectrogram, os.path.join(category_path, file)): file 
            for file in files
        }
        
        # as_completed() yields futures as they finish - more efficient than waiting for all
        for future in as_completed(future_to_file):
            file=future_to_file[future]
            try:
                #get the result from this future
                spectrogram, sample_rate=future.result()
                if spectrogram is not None:
                    spectrograms.append(spectrogram)
                    sr=sample_rate # Store the sample rate for later use
            except Exception as e:
                print(f"Error processing {file}: {e}")

    # check if we got any valid spectrograms
    if not spectrograms:
        print(f"Error: No valid spectrograms computed for category: {category}")
        return

    # compute the average spectrogram - done in main thread as it's fast
    # axis=0 means we average across all spectrograms for each time-frequency point
    avg_spectrogram=np.mean(spectrograms, axis=0)
    avg_spectrogram_db=librosa.amplitude_to_db(avg_spectrogram, ref=np.max) # convert to decibel scale for better visualization


    fig=plt.figure(figsize=(12, 6))
    # display spectrogram with logarithmic frequency scale for better detail
    librosa.display.specshow(avg_spectrogram_db, sr=sr, x_axis="time", y_axis="log", cmap="inferno") 

    plt.title(f"Average Spectrogram for {category}", fontsize=14, fontweight="bold")
    plt.xlabel("Time (s)", fontsize=12, fontweight="bold")
    plt.ylabel("Frequency (Hz)", fontsize=12, fontweight="bold")
    plt.colorbar(label="Amplitude (dB)")

    # Save the plot    
    output_path=f"../website/plots/average_spectogram"
    os.makedirs(output_path, exist_ok=True)
    plot_filename=f"spectrogram_avg_{category}.png"
    plt.savefig(f"{output_path}/{plot_filename}", dpi=300, bbox_inches="tight")
    print(f"Saved Average Spectrogram Plot: {plot_filename}")

    # explicitly close the figure to free memory
    # this is important when processing many files to prevent memory leaks
    plt.close(fig)


def plot_time_series_rms(category):
    """
    Function to generate and save a time-series plot of RMS energy for a given category.

    This function:
    - Loads the RMS energy feature data for the specified category.
    - Applies a rolling mean (window size=5) to smooth fluctuations.
    - Plots the smoothed RMS energy over time.
    - Saves the generated plot as a PNG file.

    Parameters:
    - category (str): The category name (e.g., "rain_sounds", "wind_sounds").

    Saves:
    - A time-series RMS energy plot as `time_series_rms_{category}.png` in `../website/plots/time_series_rms/`
    """

    feature_name="rms_energy"
    plt.figure(figsize=(12, 6))

    df=load_features(category)

    #take rolling mean to smooth fluctuations
    time_series=df[feature_name].rolling(window=5).mean()

    plt.plot(time_series, label=category, color="red", alpha=0.8)
    plt.title(f"Time-Series of {feature_name} for {category}", fontsize=14, fontweight="bold")
    plt.xlabel("Time (Frames)", fontsize=12, fontweight="bold")
    plt.ylabel(feature_name, fontsize=12, fontweight="bold")
    
    plt.ylim(0,0.65) #set a common ylim so that all plots have the same limit. 
    # this limit was decided by looking at the data. Adjust this limit as needed based on the data

    #save the plot
    output_path="../website/plots/time_series_rms"
    os.makedirs(output_path, exist_ok=True)
    plot_filename=f"time_series_rms_{category}.png"
    plt.savefig(f"{output_path}/{plot_filename}", dpi=300, bbox_inches="tight")
    print(f"Saved Time-Series RMS Energy Plot: {output_path}/{plot_filename}")

    plt.close()


def plot_zcr_histogram(category):
    """
    Function to generate and save a histogram of the zero-crossing rate (ZCR) for a given category.

    This function:
    - Loads the zero-crossing rate (ZCR) feature data for the specified category.
    - Checks if the feature exists in the dataset before proceeding.
    - Plots a histogram with kernel density estimation (KDE) for better visualization.
    - Saves the generated plot as a PNG file.

    Parameters:
    - category (str): The category name (e.g., "rain_sounds", "wind_sounds").

    Saves:
    - A zero-crossing rate histogram as `zcr_histogram_{category}.png` in `../website/plots/zcr_histogram/`.
    """
    
    feature_name="zero_crossing_rate"
    df=load_features(category)

    if feature_name not in df.columns:
        print(f"Feature '{feature_name}' not found in dataset for {category}. Skipping.")
        return

    plt.figure(figsize=(12, 6))

    #plot histogram
    sns.histplot(df[feature_name], bins=30, kde=True, color="salmon", alpha=0.7)

    plt.title(f"Zero-Crossing Rate Histogram for {category}", fontsize=14, fontweight="bold")
    plt.xlabel("Zero-Crossing Rate", fontsize=12, fontweight="bold")
    plt.ylabel("Frequency", fontsize=12, fontweight="bold")
    
    plt.ylim(0,210)#set a common ylim so that all plots have the same limit. 
    # this limit was decided by looking at the data. Adjust this limit as needed based on the data
    
    #save the plot
    output_path="../website/plots/zcr_histogram"
    os.makedirs(output_path, exist_ok=True)
    plot_filename=f"zcr_histogram_{category}.png"
    plt.savefig(f"{output_path}/{plot_filename}", dpi=300, bbox_inches="tight")
    print(f"Saved ZCR Histogram Plot: {output_path}/{plot_filename}")

    plt.close()


def plot_mean_spectral_contrast():
    """
    Function to generate and save a lollipop chart for the mean spectral contrast per category.

    This function:
    - Loads and combines spectral contrast feature data for all categories.
    - Computes the mean spectral contrast per category.
    - Creates a lollipop chart with distinct styling (`C1-` for lines, `C3o` for markers).
    - Adds numerical labels to each data point for clarity.
    - Saves the generated plot as a PNG file.

    Saves:
    - A lollipop chart as `mean_spectral_contrast_lollipop.png` in `../website/plots/mean_spectral_contrast/`.
    """

    df=load_combined_features()

    #extract spectral contrast columns and compute mean per category
    spectral_contrast_cols=[col for col in df.columns if "spectral_contrast" in col]
    df["mean_spectral_contrast"]=df[spectral_contrast_cols].mean(axis=1)

    #compute mean spectral contrast per category
    category_means=df.groupby("Category")["mean_spectral_contrast"].mean().sort_values()

    # plot lollipop chart
    plt.figure(figsize=(12, 7))
    plt.stem(category_means.index, category_means.values, linefmt="C1-", markerfmt="C3o", basefmt="black")

    # add labels to the dots
    for category, value in zip(category_means.index, category_means.values):
        plt.text(category, value+0.5, f"{value:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # styling
    plt.xlabel("Category", fontsize=12, fontweight="bold")
    plt.ylabel("Mean Spectral Contrast", fontsize=12, fontweight="bold")
    plt.title("Mean Spectral Contrast per Category", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.ylim(15,25)

    #save the plot
    output_path="../website/plots/mean_spectral_contrast"
    os.makedirs(output_path, exist_ok=True)
    plot_filename="mean_spectral_contrast_lollipop.png"
    plt.savefig(f"{output_path}/{plot_filename}", dpi=300, bbox_inches="tight")
    print(f"Saved Mean Spectral Contrast Lollipop Chart: {plot_filename}")

    plt.close()


def plot_spectral_rolloff():
    """
    Generate and save a violin plot of the spectral roll-off distribution across sound categories.

    This function:
    - Loads and combines spectral roll-off feature data for all categories.
    - Checks if the "spectral_rolloff" feature exists in the dataset before proceeding.
    - Creates a violin plot to visualize the distribution of spectral roll-off values.
    - Uses the "cool" colormap and highlights quartiles within each category.
    - Saves the generated plot as a PNG file.

    Saves:
    - A spectral roll-off violin plot as `spectral_rolloff_violin.png` in `../website/plots/spectral_rolloff_violin/`.
    """

    df=load_combined_features()

    if "spectral_rolloff" not in df.columns:
        print("Feature 'spectral_rolloff' not found in dataset.")
        return

    # make violin plot
    plt.figure(figsize=(12, 6))
    sns.violinplot(x="Category", y="spectral_rolloff", data=df, hue="Category", palette="cool", inner="quartile")

    plt.title("Spectral Roll-off Distribution Across Categories", fontsize=14, fontweight="bold")
    plt.xlabel("Sound Categories", fontsize=12, fontweight="bold")
    plt.ylabel("Spectral Roll-off (Hz)", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45)

    #save the plot
    output_path="../website/plots/spectral_rolloff_violin"
    os.makedirs(output_path, exist_ok=True)
    plot_filename="spectral_rolloff_violin.png"
    plt.savefig(f"{output_path}/{plot_filename}", dpi=300, bbox_inches="tight")
    print(f"Saved Spectral Roll-off Violin Plot: {output_path}/{plot_filename}")

    plt.close()


def plot_tonnetz_bar(category):
    """
    Generate and save a grouped bar chart of Tonnetz features for a given category.

    This function:
    - Loads Tonnetz feature data for the specified category.
    - Extracts the mean values for all six Tonnetz dimensions.
    - Creates a bar chart with labeled bars displaying mean values.
    - Uses the "rocket" color palette for better aesthetics.
    - Saves the generated plot as a PNG file.

    Parameters:
    - category (str): The category name (e.g., "rain_sounds", "wind_sounds").

    Saves:
    - A Tonnetz feature bar plot as `tonnetz_bar_{category}.png` in `../website/plots/tonnetz_bars/`.
    """

    df=load_features(category)

    # Extract Tonnetz columns
    tonnetz_cols=[col for col in df.columns if "tonnetz" in col]

    if not tonnetz_cols:
        print(f"Tonnetz features not found for category: {category}")
        return

    #compute mean Tonnetz values
    tonnetz_means=df[tonnetz_cols].mean()

    #create the bar plot
    plt.figure(figsize=(8, 5))
    colors=sns.color_palette("rocket", len(tonnetz_cols))
    bars=plt.bar(tonnetz_means.index, tonnetz_means.values, color=colors, alpha=0.8)

    #add value labels on top of bars
    for bar in bars:
        yval=bar.get_height()
        plt.text(bar.get_x()+bar.get_width()/2, yval, f"{yval:.5f}", 
                 ha='center', va='bottom', fontsize=9, fontweight="bold", color="black")

    #formatting
    plt.xticks(tonnetz_means.index, [f"T{i+1}" for i in range(len(tonnetz_cols))])
    plt.xlabel("Tonnetz Dimensions", fontsize=12, fontweight="bold")
    plt.ylabel("Mean Tonnetz Value", fontsize=12, fontweight="bold")
    plt.title(f"Tonnetz Feature Distribution - {category}", fontsize=14, fontweight="bold")

    #save the plot
    output_path="../website/plots/tonnetz_bars"
    os.makedirs(output_path, exist_ok=True)
    plot_filename=f"tonnetz_bar_{category}.png"
    plt.savefig(f"{output_path}/{plot_filename}", dpi=300, bbox_inches="tight")
    print(f"Saved Tonnetz bar plot: {output_path}/{plot_filename}")

    plt.close()


def plot_feature_variability(feature_name):
    """
    Generate and save a boxen plot to visualize the variability of a selected feature across sound categories.

    This function:
    - Loads and combines feature data for all categories.
    - Checks if the specified feature exists in the dataset before proceeding.
    - Plots a boxen plot to show feature variability across different categories.
    - Uses the "coolwarm" colormap for distinction.
    - Saves the generated plot as a PNG file.

    Parameters:
    - feature_name (str): The name of the feature to be analyzed (e.g., "spectral_centroid", "rms_energy").

    Saves:
    - A feature variability plot as `feature_variability_{feature_name}.png` in `../website/plots/feature_variability/`.
    """

    df=load_combined_features()

    if feature_name not in df.columns:
        print(f"Feature '{feature_name}' not found in dataset.")
        return

    #make the boxenplot
    plt.figure(figsize=(12, 6))
    sns.boxenplot(x="Category", y=feature_name, data=df, hue="Category",palette="coolwarm")

    plt.title(f"Feature Variability for {feature_name}", fontsize=14, fontweight="bold")
    plt.xlabel("Sound Categories", fontsize=12, fontweight="bold")
    plt.ylabel(feature_name, fontsize=12, fontweight="bold")
    plt.xticks(rotation=45)

    #save the plot
    plot_filename=f"feature_variability_{feature_name}.png"
    output_path="../website/plots/feature_variability"
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(f"{output_path}/{plot_filename}", dpi=300, bbox_inches="tight")
    print(f"Saved Feature Variability Plot: {output_path}/{plot_filename}")

    plt.close()

if __name__=="__main__":

    for category in categories: #for all categories make the following plots
        plot_mfcc_radial(category)
        plot_time_series_rms(category)
        plot_avg_spectrogram(category)
        plot_zcr_histogram(category)
        plot_tonnetz_bar(category)

    # then we make the next plots
    plot_3d_spectral_features()
    plot_chroma_heatmap()
    plot_mean_spectral_contrast()
    plot_spectral_rolloff()

    features=['zero_crossing_rate','spectral_centroid', 'spectral_bandwidth', 'rms_energy', 'spectral_rolloff']
    # finally for the specified features we make the feature variability plot
    for feature in features:
        plot_feature_variability(feature)