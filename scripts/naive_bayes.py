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
    
    results_visualization(X_test_discritized,y_test,y_pred,model,"multinomial_nb")

    print("Completed Multinomial Naive Bayes Classification")

def perform_gaussian_nb():
    """
    Function to train and evaluate a Gaussian Naive Bayes classifier on standardized audio features.

    This function:
    - Loads and prepares the dataset using `get_data`.
    - Scales features using StandardScaler (mean=0, std=1).
    - Trains a GaussianNB model on the training set.
    - Evaluates model performance and visualizes results.

    Returns:
    - None (Saves confusion matrix, classification heatmap, and ROC curve as PNG files).
    """

    X,y=get_data()
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=17)

    # standard scale to normalize the data
    scaler=StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    
    model=GaussianNB()
    model.fit(X_train_scaled,y_train)
    y_pred=model.predict(X_test_scaled)

    results_visualization(X_test_scaled,y_test,y_pred,model,"gaussian_nb")

    print("Completed Gaussian Naive Bayes Classification")

def perform_bernoulli_nb():
    """
    Function to train and evaluate a Bernoulli Naive Bayes classifier on binarized audio features.

    This function:
    - Loads and prepares the dataset using `get_data`.
    - Binarizes features: 1 if greater than the mean of each feature, else 0.
    - Trains a BernoulliNB model on the training set.
    - Evaluates model performance and visualizes results.

    Returns:
    - None (Saves confusion matrix, classification heatmap, and ROC curve as PNG files).
    """

    X,y=get_data()
    feature_means = np.mean(X, axis=0)

    # we convert the features to binary by making them equal to 1 if the value exceeds mean, and 0 otherwise
    X_binary = (X>feature_means).astype(int) 

    X_train,X_test,y_train,y_test=train_test_split(X_binary,y,test_size=0.3,random_state=17)
    
    model=BernoulliNB()
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)

    results_visualization(X_test,y_test,y_pred,model,"bernoulli_nb")
    
    print("Completed Bernoulli Naive Bayes Classification")

def results_visualization(X_test,y_test,y_pred,model,model_name):
    """
    Function to visualize model performance through confusion matrix, classification report heatmap, and ROC curve.

    This function:
    - Plots and saves a confusion matrix for predicted vs. actual labels.
    - Generates a classification report heatmap (precision, recall, F1-score).
    - Plots the ROC curve and computes AUC score.

    Parameters:
    - X_test (np.ndarray or pd.DataFrame): Test feature set used for predictions.
    - y_test (array-like): True labels for the test set.
    - y_pred (array-like): Predicted labels by the model.
    - model (sklearn model): Trained classification model (must support `predict_proba`).
    - model_name (str): Identifier for saving plots (e.g., 'multinomial_nb').

    Returns:
    - None (Saves all visualizations as PNG files in `../website/plots/naive_bayes/`).
    """
    
    output_path="../website/plots/naive_bayes"
    os.makedirs(output_path,exist_ok=True)

    figname=model_name.split("_")
    figname[0]=figname[0].capitalize()
    figname[1]=figname[1].upper()
    figname="".join(figname)

    # confusion matrix 
    cm=confusion_matrix(y_test, y_pred)
    disp=ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix - {figname}", fontsize=14, fontweight='bold')
    plt.savefig(f"{output_path}/{model_name}_confusion_matrix.png", dpi=200, bbox_inches='tight')
    plt.close()

    # classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).T
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_report.iloc[:-1, :-1], annot=True, cmap="coolwarm", fmt=".3f")
    
    plt.title(f"Classification Report Heatmap - {figname}", fontsize=14, fontweight='bold')
    plt.savefig(f"{output_path}/{model_name}_classification_report_heatmap.png", dpi=200, bbox_inches='tight')
    plt.close()

    # roc curve
    y_prob=model.predict_proba(X_test)[:, 1]
    fpr,tpr,_=roc_curve(y_test, y_prob)
    roc_auc=auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='royalblue', linewidth=2.5, label=f'AUC = {roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1.5)

    plt.xlabel("False Positive Rate", fontsize=12, fontweight='bold')
    plt.ylabel("True Positive Rate", fontsize=12, fontweight='bold')
    plt.title(f"ROC Curve - {figname}", fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()

    plt.savefig(f"{output_path}/{model_name}_roc_curve.png", dpi=200, bbox_inches='tight')
    plt.close()

if __name__=="__main__":
    perform_multinomial_nb()
    perform_gaussian_nb()
    perform_bernoulli_nb()