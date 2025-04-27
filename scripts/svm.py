import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA

def get_data():
    """
    Function to load, label, and prepare audio feature data for SVM classification.

    This function:
    - Loads audio feature CSVs for two categories: 'laughter' and 'footsteps'.
    - Combines them into a single DataFrame.
    - Encodes 'laughter' as 0 and 'footsteps' as 1 for binary classification.
    - Shuffles the dataset and separates features (X) and labels (y).

    Returns:
    - X (pd.DataFrame): Feature set excluding the label column.
    - y (pd.Series): Binary target labels (0 for laughter, 1 for footsteps).
    """
    
    file_path="../audiofiles_processed_features_CSVs"
    categories=["footsteps","laughter"] #take laughter and footsteps sounds
    df=pd.concat([pd.read_csv(f"{file_path}/{category}_features.csv") for category in categories])
    
    df=df.sample(frac=1,random_state=17).reset_index(drop=True)
    df.drop("filename",axis=1,inplace=True)
    # encode laughter as 0 and footsteps as 1
    df["category"]=df["category"].apply(lambda x: 0 if x=="laughter" else 1)

    y=df["category"]
    X=df.drop("category",axis=1)

    return X,y

def perform_support_vector_machine_classification(kernel,C,degree=3):
    """
    Function to train and evaluate a Support Vector Machine (SVM) classifier.

    This function:
    - Loads and splits the dataset into training and testing sets.
    - Applies StandardScaler to normalize the features.
    - Trains an SVM model with specified kernel and hyperparameters.
    - Evaluates the model and visualizes results (confusion matrix, classification heatmap, ROC curve, decision boundary).

    Parameters:
    - kernel (str): Kernel type for SVM ('linear', 'rbf', or 'poly').
    - C (float): Regularization parameter.
    - degree (int, optional): Degree for polynomial kernel (default is 3).

    Returns:
    - None (Saves plots under '../website/plots/svm/').
    """

    X,y=get_data()

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=17)

    # standard scale to normalize the data
    scaler=StandardScaler()
    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)
    
    #if kernel=poly we add an extra parameter degree=3 to our computations
    if kernel=="poly":
        model=SVC(kernel=kernel,C=C,degree=degree,probability=True)
        model.fit(X_train_scaled,y_train)
        y_pred=model.predict(X_test_scaled)
        results_visualization(X_test_scaled,y_test,y_pred,model,f"SVM kernel={kernel} degree={degree} C={C}")
        plot_decision_boundary(model, X_test_scaled, y_test, kernel, C)

        print(f"Completed SVM Classification for {kernel} kernel degree={degree} and C={C}")
    else: #otherwise we dont add degree
        model=SVC(kernel=kernel,C=C,probability=True)
        model.fit(X_train_scaled,y_train)
        y_pred=model.predict(X_test_scaled)
        results_visualization(X_test_scaled,y_test,y_pred,model,f"SVM kernel={kernel} C={C}")
        plot_decision_boundary(model, X_test_scaled, y_test, kernel, C, degree)

        print(f"Completed SVM Classification for {kernel} kernel and C={C}")


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

    output_path="../website/plots/svm"
    os.makedirs(output_path,exist_ok=True)

    filename=model_name.replace("=","_")
    filename=filename.lower()
    filename=filename.replace(" ","_")

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


def plot_decision_boundary(trained_model, X, y, kernel, C, degree=3):
    """
    Function to visualize the decision boundary of a trained SVM model using 2D PCA-reduced data.

    This function:
    - Reduces feature dimensions to 2D using PCA.
    - Creates a mesh grid covering the 2D feature space.
    - Predicts labels for each point in the grid using the SVM model.
    - Plots the decision boundary and actual data points.
    - Saves the decision boundary plot.

    Parameters:
    - trained_model: Trained SVM model (must have `predict` method).
    - X (np.ndarray or pd.DataFrame): Full feature set.
    - y (array-like): True labels corresponding to X.
    - kernel (str): Kernel type used ('linear', 'rbf', 'poly').
    - C (float): Regularization parameter used.
    - degree (int, optional): Degree of polynomial kernel if applicable (default is 3).

    Returns:
    - None (Saves decision boundary plot as PNG file).
    """

    #reduce dimensionality to 2D using PCA for visualization
    reducer=PCA(n_components=2)
    X_reduced=reducer.fit_transform(X)
    
    #create a mesh grid for the reduced space
    x_min, x_max=X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
    y_min, y_max=X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
    xx, yy=np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
    
    #map grid points back to original feature space using inverse transform
    grid_points=np.c_[xx.ravel(), yy.ravel()]
    grid_points_original=reducer.inverse_transform(grid_points)
    #predict class for each point in the mesh grid
    Z=trained_model.predict(grid_points_original)
    Z=Z.reshape(xx.shape)

    #plot title and save file name based on kernel
    if kernel=="poly":
        title=f"Kernel={kernel} degree={degree} C={C}"
        filename=f"svm_kernel_{kernel}_degree_{degree}_c_{C}_decision_boundary"
    else:
        title=f"Kernel={kernel} C={C}"
        filename=f"svm_kernel_{kernel}_c_{C}_decision_boundary"

    #plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx,yy,Z,alpha=0.5,cmap='coolwarm')
    plt.scatter(X_reduced[:,0], X_reduced[:,1],c=y,edgecolors='k',cmap='coolwarm')
    plt.xlabel(f"PC1",fontsize=12)
    plt.ylabel(f"PC2",fontsize=12)
    plt.title(f"Decision Boundary - {title}",fontsize=14,fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    handles=[plt.Line2D([0],[0],marker='o',color='w',markerfacecolor=plt.cm.coolwarm(0),markersize=10),
             plt.Line2D([0],[0],marker='o',color='w',markerfacecolor=plt.cm.coolwarm(255), markersize=10)]
    labels=['Class 0', 'Class 1']
    plt.legend(handles, labels, loc='best', fontsize=11)
    
    #save the plot
    output_path="../website/plots/svm"
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(f"{output_path}/{filename}.png", dpi=200, bbox_inches='tight')
    plt.close()


if __name__=="__main__":
    """
    Trains and evaluates SVM models for three kernels ('linear', 'rbf', 'poly') 
    using three different regularization parameters (C values: 0.5, 10, 100).
    """
    for kernel in ["linear","rbf","poly"]:
        for cost in [0.5,10,100]:
            perform_support_vector_machine_classification(kernel,cost)
            