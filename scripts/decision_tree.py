import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random

def get_data():
    """
    Function to load, label, and prepare audio feature data for classification.

    This function:
    - Loads audio feature CSVs for birds and dog_bark sound categories.
    - Combines them into a single DataFrame.
    - Maps original categories into two classes: 'birds' (1) and 'dog_bark' (0).
    - Shuffles the data and separates features (X) and target labels (y).

    Returns:
    - X (pd.DataFrame): Feature set excluding the label column.
    - y (pd.Series): Binary target labels (1 for birds, 0 for dog_bark).
    """
    
    file_path="../audiofiles_processed_features_CSVs"
    categories=["birds","dog_bark"] #take bird sounds and dog_bark sounds
    df=pd.concat([pd.read_csv(f"{file_path}/{category}_features.csv") for category in categories])
    
    df=df.sample(frac=1,random_state=17).reset_index(drop=True)
    df.drop("filename",axis=1,inplace=True)
    # encode birds as 1 and dog_bark as 0
    df["category"]=df["category"].apply(lambda x: 1 if x=="birds" else 0)

    y=df["category"]
    X=df.drop("category",axis=1)

    return X,y

def perform_decision_tree_classification():
    """
    Function to train and evaluate multiple Decision Tree classifiers with different configurations.

    This function:
    - Loads and prepares the dataset using `get_data`.
    - Shuffles feature columns to simulate randomized selection.
    - Trains three decision trees with different hyperparameters:
        - Tree 1: Gini impurity, max depth = 3, max features = 10
        - Tree 2: Log loss (cross-entropy), max depth = 2, max features = 5
        - Tree 3: Entropy, max depth = 3
    - Evaluates each tree using `plot_and_evaluate_tree` and saves performance visualizations.

    Returns:
    - None (Prints progress and saves all evaluation plots for each model)
    """

    X, y=get_data()
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=18)

    scaler=StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)

    # we preserve feature names for plotting the tree
    features=list(X.columns)
    X_train_scaled=pd.DataFrame(X_train_scaled, columns=features)
    X_test_scaled=pd.DataFrame(X_test_scaled, columns=features)

    random.seed(17)
    random.shuffle(features)

    #Tree 1: Gini (depth=3, features=10)")
    tree1=DecisionTreeClassifier(max_depth=3, max_features=10,criterion="gini", random_state=18)
    plot_and_evaluate_tree(X_train_scaled[features], X_test_scaled[features], y_train, y_test, tree1, "Tree 1: Gini (depth=3, features=10)")
    print("Completed Tree 1")

    #Tree 2: LogLoss (depth=2, features=5)
    tree2=DecisionTreeClassifier(max_depth=2, criterion="log_loss", max_features=5, random_state=18)
    plot_and_evaluate_tree(X_train_scaled[features], X_test_scaled[features], y_train, y_test, tree2, "Tree 2: LogLoss (depth=2, features=5)")
    print("Completed Tree 2")

    #Tree 3: Entropy (depth=3)
    tree3=DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=18)
    plot_and_evaluate_tree(X_train_scaled, X_test_scaled, y_train, y_test, tree3, "Tree 3: Entropy (depth=3)")
    print("Completed Tree 3")

def plot_and_evaluate_tree(X_train, X_test, y_train, y_test, model, title):
    """
    Function to train a Decision Tree classifier and visualize its performance.

    This function:
    - Fits the given Decision Tree model to training data.
    - Generates and saves four types of evaluation plots:
        - Confusion Matrix
        - Classification Report Heatmap
        - Decision Tree Visualization
        - ROC Curve with AUC score

    Parameters:
    - X_train (pd.DataFrame): Training feature set.
    - X_test (pd.DataFrame): Test feature set.
    - y_train (pd.Series): Training target labels.
    - y_test (pd.Series): Test target labels.
    - model (DecisionTreeClassifier): The decision tree model to train and evaluate.
    - title (str): Title used for plot labeling and filenames.

    Returns:
    - None (Saves all plots as PNG files in `../website/plots/decision_trees/`)
    """

    output_path="../website/plots/decision_trees"
    os.makedirs(output_path,exist_ok=True)

    model.fit(X_train, y_train)
    y_pred=model.predict(X_test)

    # confusion matrix 
    cm=confusion_matrix(y_test, y_pred)
    disp=ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix - {title}", fontsize=11, fontweight='bold')
    plt.savefig(f"{output_path}/tree_{title.split()[1][:1]}_confusion_matrix.png", dpi=200, bbox_inches='tight')
    plt.close()

    # classification report
    report=classification_report(y_test, y_pred, output_dict=True)
    df_report=pd.DataFrame(report).T
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_report.iloc[:-1, :-1], annot=True, cmap="coolwarm", fmt=".3f")

    plt.title(f"Classification Report Heatmap - {title}", fontsize=11, fontweight='bold')
    plt.savefig(f"{output_path}/tree_{title.split()[1][:1]}_classification_report.png", dpi=200, bbox_inches='tight')
    plt.close()

    # plot tree
    plt.figure(figsize=(14, 6))
    plot_tree(model, feature_names=X_train.columns, class_names=["dog_bark", "birds"], filled=True)
    plt.title(f"Decision Tree - {title}", fontsize=14, fontweight='bold')
    plt.savefig(f"{output_path}/tree_{title.split()[1][:1]}_tree_plot.png", dpi=200, bbox_inches='tight')
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
    plt.title(f"ROC Curve - {title}", fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{output_path}/tree_{title.split()[1][:1]}_roc_curve.png", dpi=200, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    perform_decision_tree_classification()