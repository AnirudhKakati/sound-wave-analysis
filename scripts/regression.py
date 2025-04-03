import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os

def get_data():
    """
    Function to load, label, and prepare audio feature data for classification.

    This function:
    - Loads audio feature CSVs for selected sound categories.
    - Combines them into a single DataFrame.
    - Maps original categories into two classes: 'music' (1) and 'outdoor' (0).
    - Shuffles the data and separates features (X) and target labels (y).

    Returns:
    - X (pd.DataFrame): Feature set excluding the label column.
    - y (pd.Series): Binary target labels (1 for music, 0 for outdoor).
    """
    
    file_path="../audiofiles_processed_features_CSVs"
    categories=["piano","guitar","drums","thunder_sounds","rain_sounds","wind_sounds"] #take music sounds and outdoor sounds
    df=pd.concat([pd.read_csv(f"{file_path}/{category}_features.csv") for category in categories])
    
    df=df.sample(frac=1,random_state=17).reset_index(drop=True)
    df.drop("filename",axis=1,inplace=True)
    # convert them to two categories, music and outdoor
    df["category"]=df["category"].apply(lambda x:"music" if x in ("guitar","piano","drums") else "outdoor")
    # encode music as 1 and outdoor as 0
    df["category"]=df["category"].apply(lambda x: 0 if x=="outdoor" else 1)

    y=df["category"]
    X=df.drop("category",axis=1)

    return X,y

def perform_multinomial_nb():
    """
    Function to train and evaluate a Multinomial Naive Bayes classifier on discretized audio features.

    This function:
    - Loads and prepares the dataset using `get_data`.
    - Scales features to [0, 1] using MinMaxScaler.
    - Discretizes the scaled features into integers (0–100).
    - Trains a MultinomialNB model on the training set.
    - Evaluates model performance and visualizes results.

    Returns:
    - None (Saves confusion matrix, classification heatmap, and ROC curve as PNG files).
    """
    
    X,y=get_data()
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=17)

    # to discretize we will first min max scale to make the numbers betwen 0 to 1, then multiply them by 100 and convert to integers
    scaler=MinMaxScaler() 
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)

    X_train_discritized=np.round(X_train_scaled*100).astype(int) 
    X_test_discritized=np.round(X_test_scaled*100).astype(int)

    model = MultinomialNB()
    model.fit(X_train_discritized, y_train)
    y_pred = model.predict(X_test_discritized)
    
    # results_visualization(X_test_discritized,y_test,y_pred,model,"multinomial_nb")

    print("Completed Multinomial Naive Bayes Classification")

