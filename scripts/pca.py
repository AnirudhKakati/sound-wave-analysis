import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import json
import os

with open("../scripts/categories.txt", "r") as f: #load categories
    categories=f.read().split()

def perform_pca():
    """
    Function to prepare dataset and perform PCA.

    This function:
    - Reads extracted feature CSVs for all audio categories and concatenates them.
    - Drops non-numeric columns such as "filename" and the label "category".
    - Standardizes the feature data using `StandardScaler`.
    - Applies Principal Component Analysis (PCA) to transform the dataset.

    Parameters:
    - None

    Returns:
    - x_pca (ndarray): PCA-transformed feature data.
    - pca (PCA object): Fitted PCA model.
    - df (DataFrame): Original dataset before PCA transformation.
    """

    # combine all csvs into one dataframe
    dfs=[pd.read_csv(f"../audiofiles_processed_features_CSVs/{category}_features.csv") for category in categories]
    df=pd.concat(dfs, ignore_index=True)    

    df_features=df.drop(["filename", "category"], axis=1) #drop the categorical column and the label

    scaler=StandardScaler() #scale the dataset
    x_scaled=scaler.fit_transform(df_features)

    pca=PCA() #perform pca will all components
    x_pca=pca.fit_transform(x_scaled)

    return x_pca, pca, df

def prepare_pca_for_web():
    """
    Function to prepare and save PCA data for web visualization.

    This function:
    - Loads extracted feature CSVs for all categories and concatenates them.
    - Standardizes the feature data using `StandardScaler`.
    - Performs PCA to reduce dimensionality to 3 principal components.
    - Creates a dictionary storing PCA-transformed values for each category.
    - Saves the PCA data as a JSON file for web visualization.
    - Computes the number of principal components required to explain at least 95% variance.
    - Prints the top 3 eigenvalues.

    Parameters:
    - None

    Returns:
    - None (Outputs a JSON file containing PCA results and prints variance details).
    """

    x_pca, pca, df=perform_pca()
    x_pca=x_pca[:,:3]
    
    df_pca=pd.DataFrame(x_pca, columns=["PC1", "PC2", "PC3"]) # 3d pca
    df_pca["category"]=df["category"] # add category information to the df
    
    pca_data={}
    
    for category in categories: # for each category
        category_df=df_pca[df_pca["category"] == category] # filter the PCA-transformed data for the current category
        
        # store PCA values for each principal component in this category
        pca_data[category]={
            "PC1": category_df["PC1"].tolist(),
            "PC2": category_df["PC2"].tolist(),
            "PC3": category_df["PC3"].tolist(),
            "count": len(category_df)
        }
    
    os.makedirs("../website/data", exist_ok=True)
    with open("../website/data/pca_data.json", "w") as f: # save the PCA data as JSON
        json.dump(pca_data, f)
    
    print(f"PCA data for web visualization saved to '../website/data/pca_data.json'")

    req_comp=0 #the req no. of components for 95% variance
    cumulative_variance=np.cumsum(pca.explained_variance_ratio_) 
    for i,c in enumerate(cumulative_variance):
        if c>0.95: #for the component where cumulative variance crosses 95%
            req_comp=i+1 #that is the required component
            break
    print("\nRequired no. of Principle Components to explain atleast 95% variance :",req_comp, "\nExplained variance :", cumulative_variance[req_comp-1])
    
    
    plt.figure(figsize=(10, 6)) # we create scree plot to visualize cumulative explained variance
    plt.bar(range(1,len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_,alpha=0.6,color='skyblue',label='Individual Explained Variance')
    
    plt.step(range(1,len(cumulative_variance)+1), cumulative_variance, where='mid', color='red', label='Cumulative Explained Variance')
    
    plt.axhline(y=0.95, color='green', linestyle='--', alpha=0.8, label='95% Threshold') # the 95% threshold is highlighted
    plt.axvline(x=req_comp, color='purple', linestyle='--', alpha=0.8, label=f'Components Needed: {req_comp}') # corresponding number of components is also marked
    
    #text annotation for the 95% threshold point
    plt.scatter(req_comp, cumulative_variance[req_comp-1], color='purple', s=100, zorder=5) 
    plt.annotate(f'({req_comp}, {cumulative_variance[req_comp-1]:.3f})', xy=(req_comp, cumulative_variance[req_comp-1]),
                 xytext=(req_comp + 1, cumulative_variance[req_comp-1] + 0.05), arrowprops=dict(facecolor='black', shrink=0.05, width=1.5), fontsize=10)
    
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Components (Scree Plot)')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # save the plot
    output_path=f"../website/plots/pca"
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(f"{output_path}/scree_plot.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nScree plot saved to '{output_path}/scree_plot.png'")
    


    top_3_eigen_values=list(pca.explained_variance_[:3]) #get the top 3 eigen values
    print(f"\nTop 3 Eigen Values : {top_3_eigen_values[0]}, {top_3_eigen_values[1]}, {top_3_eigen_values[2]}")

    plt.figure(figsize=(8, 5)) # bar chart for top 3 eigenvalues
    bars = plt.bar(['PC1', 'PC2', 'PC3'], top_3_eigen_values, color=['#ff9999', '#66b3ff', '#99ff99'])
    
    # add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Top 3 Eigenvalues')
    plt.ylabel('Eigenvalue')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # save the plot
    output_path=f"../website/plots/pca"
    plt.savefig(f"{output_path}/top_eigenvalues.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nTop eigenvalues plot saved to '{output_path}/top_eigenvalues.png'")


def pca_visualization_2d_3d():
    """
    Function to generate and save 2D and 3D PCA visualizations.

    This function:
    - Performs PCA on the dataset, reducing it to 3 components.
    - Generates a 2D scatter plot using the first two principal components.
    - Computes and displays the explained variance percentage for 2D PCA.
    - Saves the 2D PCA plot as a PNG file.
    - Generates a 3D scatter plot using the first three principal components.
    - Computes and displays the explained variance percentage for 3D PCA.
    - Saves the 3D PCA plot as a PNG file.
    - Calls `visualize_feature_contributions` to generate a heatmap of feature contributions.

    Parameters:
    - None

    Returns:
    - None (Outputs PCA visualization plots as PNG files).
    """

    x_pca, pca, df=perform_pca() #this returns pca will all components
    x_pca=x_pca[:,:3] #we need upto 3 components for 2d and 3d pca
    df_pca=pd.DataFrame(x_pca,columns=[f"PC{i+1}" for i in range(3)]) #convert to a dataframe

    # plot 2d pca
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df_pca["PC1"], y=df_pca["PC2"], alpha=0.6, color="salmon") #take only pc1 and pc2 columns
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("2D PCA Projection of Audio Features")

    explained_variance_2d=sum(pca.explained_variance_ratio_[:2])*100 #calculate the explained variance % for 2d
    # add that text to the plot
    plt.text(x=df_pca["PC1"].min(), y=df_pca["PC2"].max(), s=f"Explained Variance %: {explained_variance_2d:.3f}%", bbox=dict(facecolor='white', alpha=0.6))
    
    #save the 2d plot
    output_path=f"../website/plots/pca"
    os.makedirs(output_path,exist_ok=True)
    plot_filename=f"pca_2d.png"
    plt.savefig(f"{output_path}/{plot_filename}", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n2D PCA Projection of Audio Features saved to '{output_path}/{plot_filename}'")

    #plot 3d pca
    fig=plt.figure(figsize=(10, 7))
    ax=fig.add_subplot(111, projection='3d')
    ax.scatter(df_pca["PC1"], df_pca["PC2"], df_pca["PC3"], alpha=0.6, color="cyan", edgecolors="black") #take all 3 pc columns
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")
    ax.set_title("3D PCA Projection of Audio Features")

    explained_variance_3d=sum(pca.explained_variance_ratio_[:3])*100#calculate the explained variance % for 2d
    # add that text to the plot
    ax.text(df_pca["PC1"].min(), df_pca["PC2"].min(), 2*df_pca["PC3"].max(), s=f"Explained Variance %: {explained_variance_3d:.3f}%", 
            bbox=dict(facecolor='white', alpha=0.6))
    
    #save the 3d plot
    plot_filename="pca_3d.png"
    plt.savefig(f"{output_path}/{plot_filename}", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n3D PCA Projection of Audio Features saved to '{output_path}/{plot_filename}'")
    visualize_feature_contributions(pca, output_path)

def visualize_feature_contributions(pca, output_path):
    """
    Function to visualize how original features contribute to the principal components.
    
    This function:
    - Creates a heatmap showing the loading/contribution of each original feature to the first three principal components.
    - Selects the top 20 features with the highest absolute contribution to PC1 for better visualization.
    - Saves the visualization as a heatmap plot in PNG format.

    Parameters:
    - pca (PCA object): Fitted PCA model containing principal component loadings.
    - output_path (str): Directory path to save the output plot.

    Returns:
    - None (Outputs a heatmap plot as a PNG file).
    """
    # we gett the components and feature names
    components=pca.components_ 
    sample_df=pd.read_csv(f"../audiofiles_processed_features_CSVs/{categories[0]}_features.csv")
    feature_names=sample_df.drop(["filename", "category"], axis=1).columns.tolist()
    
    # features are sorted by absolute contribution to PC1 for better visualization
    pc1_importance=np.abs(components[0])
    sorted_indices=np.argsort(pc1_importance)[::-1]
    top_n=min(20, len(feature_names))  # showing top 20 features or all if less
    
    # get top contributing features
    top_features=[feature_names[i] for i in sorted_indices[:top_n]]
    
    
    # heatmap of top feature contributions to all three PCs
    plt.figure(figsize=(12, 10))
    heatmap_data = pd.DataFrame(components[:3, sorted_indices[:top_n]].T, index=top_features, columns=['PC1', 'PC2', 'PC3'])
    sns.heatmap(heatmap_data, cmap='coolwarm', center=0, annot=True, fmt=".2f", linewidths=.5, cbar_kws={"shrink": 0.8, "label": "Component Loading"})
    
    plt.title('Feature Contributions to Principal Components', fontsize=14, pad=20)
    plt.ylabel('Audio Features', fontsize=12)
    plt.xlabel('Principal Components', fontsize=12)
    plt.tight_layout()
    
    #save the plot
    plot_filename = "feature_contributions_heatmap.png"
    plt.savefig(f"{output_path}/{plot_filename}", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nHeatmap of Feature Contributions saved to '{output_path}/{plot_filename}'")

if __name__ == "__main__":
    prepare_pca_for_web()
    pca_visualization_2d_3d()