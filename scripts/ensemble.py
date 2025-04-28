import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os

def get_data():
    """
    Function to load, label, and prepare audio feature data for SVM classification.

    This function:
    - Loads audio feature CSVs for two categories: 'speech' and 'crowd_noise'.
    - Combines them into a single DataFrame.
    - Encodes 'speech' as 0 and 'crowd_noise' as 1 for binary classification.
    - Shuffles the dataset and separates features (X) and labels (y).

    Returns:
    - X (pd.DataFrame): Feature set excluding the label column.
    - y (pd.Series): Binary target labels (0 for speech, 1 for crowd_noise).
    """
    
    file_path="../audiofiles_processed_features_CSVs"
    categories=["crowd_noise","speech"] #take speech and crowd_noise sounds
    df=pd.concat([pd.read_csv(f"{file_path}/{category}_features.csv") for category in categories])
    
    df=df.sample(frac=1,random_state=17).reset_index(drop=True)
    df.drop("filename",axis=1,inplace=True)
    # encode speech as 0 and crowd_noise as 1
    df["category"]=df["category"].apply(lambda x: 0 if x=="speech" else 1)

    y=df["category"]
    X=df.drop("category",axis=1)

    return X,y

def perform_random_forest():
    """
    Function to train and evaluate a Random Forest classifier on speech and crowd noise audio features.

    This function:
    - Loads the preprocessed dataset using `get_data`.
    - Splits the dataset into training and testing sets (70%-30%).
    - Trains a Random Forest model with default parameters and a fixed random seed.
    - Predicts on the test set.
    - Visualizes model performance using confusion matrix, classification report, and ROC curve.

    Returns:
    - None (Saves visualizations under '../website/plots/ensemble/').
    """

    X,y=get_data()
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=17)

    model=RandomForestClassifier(random_state=17)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)

    results_visualization(X_test,y_test,y_pred,model,"RandomForest")

def results_visualization(X_test,y_test,y_pred,model,model_name):
    """
    Function to visualize model evaluation results.

    This function:
    - Plots and saves the confusion matrix.
    - Generates and saves a heatmap of the classification report.
    - Plots and saves the ROC curve.

    Parameters:
    - X_test (np.ndarray or pd.DataFrame): Feature set for testing.
    - y_test (array-like): True labels for test data.
    - y_pred (array-like): Predicted labels by the model.
    - model (sklearn SVM model): Trained SVM model.
    - model_name (str): Name of the model for saving plots.

    Returns:
    - None (Saves all visualizations as PNG files).
    """

    output_path="../website/plots/ensemble"
    os.makedirs(output_path,exist_ok=True)

    filename=model_name.lower()

    # confusion matrix 
    cm=confusion_matrix(y_test, y_pred)
    disp=ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight='bold')
    plt.savefig(f"{output_path}/{filename}_confusion_matrix.png", dpi=200, bbox_inches='tight')
    plt.close()

    # classification report
    report=classification_report(y_test, y_pred, output_dict=True)
    df_report=pd.DataFrame(report).T
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_report.iloc[:-1, :-1], annot=True, cmap="coolwarm", fmt=".3f")
    
    plt.title(f"Classification Report Heatmap - {model_name}", fontsize=14, fontweight='bold')
    plt.savefig(f"{output_path}/{filename}_classification_report_heatmap.png", dpi=200, bbox_inches='tight')
    plt.close()

    # roc curve
    y_prob=model.predict_proba(X_test)[:, 1]
    fpr,tpr,_=roc_curve(y_test, y_prob)
    roc_auc=auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='royalblue', linewidth=2.5, label=f'AUC={roc_auc:.3f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1.5)

    plt.xlabel("False Positive Rate", fontsize=12, fontweight='bold')
    plt.ylabel("True Positive Rate", fontsize=12, fontweight='bold')
    plt.title(f"ROC Curve - {model_name}", fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()

    plt.savefig(f"{output_path}/{filename}_roc_curve.png", dpi=200, bbox_inches='tight')
    plt.close()

if __name__=="__main__":
    perform_random_forest()