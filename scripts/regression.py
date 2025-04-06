import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os

def get_data():
    """
    Function to load, label, and prepare audio feature data for classification.

    This function:
    - Loads audio feature CSVs for two categories: 'fireworks' and 'sirens'.
    - Combines them into a single DataFrame.
    - Encodes 'fireworks' as 1 and 'sirens' as 0 for binary classification.
    - Shuffles the data and separates features (X) and labels (y).

    Returns:
    - X (pd.DataFrame): Feature set excluding the label column.
    - y (pd.Series): Binary target labels (1 for fireworks, 0 for sirens).
    """
    
    file_path="../audiofiles_processed_features_CSVs"
    categories=["fireworks","sirens"] #take music sirens and fireworks sounds
    df=pd.concat([pd.read_csv(f"{file_path}/{category}_features.csv") for category in categories])
    
    df=df.sample(frac=1,random_state=17).reset_index(drop=True)
    df.drop("filename",axis=1,inplace=True)
    # encode sirens as 0 and fireworks as 1
    df["category"]=df["category"].apply(lambda x: 0 if x=="sirens" else 1)

    y=df["category"]
    X=df.drop("category",axis=1)

    return X,y

def perform_logistic_regression_and_multinomial_nb():
    """
    Function to train and evaluate both Logistic Regression and Multinomial Naive Bayes classifiers.

    This function:
    - Loads and prepares the dataset using `get_data`.
    - Applies MinMax scaling to transform features into the [0, 1] range.
    - Discretizes the scaled features by multiplying by 100 and converting to integers.
    - Trains and evaluates:
        - Logistic Regression model
        - Multinomial Naive Bayes model
    - Visualizes the performance of each model using `results_visualization`.

    Returns:
    - None (Saves confusion matrix, classification heatmap, and ROC curve for each model as PNG files).
    """
    
    X,y=get_data()
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=17)

    # to discretize we will first min max scale to make the numbers betwen 0 to 1, then multiply them by 100 and convert to integers
    scaler=MinMaxScaler() 
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)

    X_train_discritized=np.round(X_train_scaled*100).astype(int) 
    X_test_discritized=np.round(X_test_scaled*100).astype(int)

    #perform logistic regression
    model = LogisticRegression()
    model.fit(X_train_discritized, y_train)
    y_pred = model.predict(X_test_discritized)
    
    results_visualization(X_test_discritized,y_test,y_pred,model,"logistic_regression")
    print("Completed Logistic Regression")

    #perform multinomial nb to compare it with
    model = MultinomialNB()
    model.fit(X_train_discritized, y_train)
    y_pred = model.predict(X_test_discritized)

    results_visualization(X_test_discritized,y_test,y_pred,model,"multinomial_nb")
    print("Completed Multinomial Naive Bayes Classification")


def results_visualization(X_test,y_test,y_pred,model,model_name):
    """
    Function to visualize model performance through various plots.

    This function:
    - Plots and saves the following:
        - Confusion matrix comparing predicted and actual labels.
        - Classification report heatmap showing precision, recall, and F1-scores.
        - ROC curve with AUC score.

    Parameters:
    - X_test (np.ndarray or pd.DataFrame): Test feature set used for predictions.
    - y_test (array-like): True labels for the test set.
    - y_pred (array-like): Predicted labels by the model.
    - model (sklearn model): Trained classification model (must support `predict_proba`).
    - model_name (str): Identifier for saving plots (e.g., 'logistic_regression', 'multinomial_nb').

    Returns:
    - None (Saves all plots as PNG files in `../website/plots/logistic_regression/`).
    """

    output_path="../website/plots/logistic_regression"
    os.makedirs(output_path,exist_ok=True)

    figname=" ".join(model_name.split("_")).title()
    if model_name=="multinomial_nb":
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
    y_prob=model.predict_proba(X_test)[:,1]
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
    perform_logistic_regression_and_multinomial_nb()

