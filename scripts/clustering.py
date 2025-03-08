import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import random
from kneed import KneeLocator

with open("../scripts/categories.txt","r") as f:
    categories=f.read().split()

def get_3d_pca_data():
    with open("../website/data/pca_data.json","r") as f:
        data=json.load(f)

    expanded_data=[]
    for category, values in data.items():
        for i in range(len(values["PC1"])):
            expanded_data.append({
                "category": category,
                "PC1": values["PC1"][i],
                "PC2": values["PC2"][i],
                "PC3": values["PC3"][i]
            })
    df_pca=pd.DataFrame(expanded_data)
    return df_pca

def perform_kmeans(samples_per_cluster=50):
    
    #we perform kmeans on the 3D PCA dataset
    df_pca=get_3d_pca_data()
    categories_array=df_pca["category"].values # store the categories for later use
    df_features=df_pca.drop("category", axis=1)
    
    silhouette_scores=[]
    K_range=range(2,11) # we try 2 to 10 clusters
    for k in K_range: 
        #perform k-means with the range of possible K values and get the silhouette scores 
        kmeans=KMeans(n_clusters=k, n_init=10, random_state=17) #take 10 different centroid initializations
        labels=kmeans.fit_predict(df_features)
        score=silhouette_score(df_features, labels)
        silhouette_scores.append((score, k)) #append the score and the k-value

    top_3_k=[k for _, k in sorted(silhouette_scores, reverse=True)[:3]] #get the top 3 values with the highest sillhouette scores
    print(f"Top 3 k values based on silhouette score: {top_3_k}")
    
    output_path="../website/plots/kmeans" #output path to save plots
    os.makedirs(output_path,exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(K_range, [x[0] for x in silhouette_scores], marker="o")
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Method for Optimal K")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    #save plot
    plot_filename="kmeans_silhouette_scores.png"   
    plt.savefig(f"{output_path}/{plot_filename}", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"KMeans Silhouette Scores Plot saved to {output_path}/{plot_filename}")
    
    X_pca=df_features.values  # we convert it to a NumPy array for convenience

    for i, k in enumerate(top_3_k):
        # perform KMeans for the top 3 k values
        kmeans=KMeans(n_clusters=k, n_init=10, random_state=17)
        clusters=kmeans.fit_predict(X_pca)
        centroids=kmeans.cluster_centers_
        
        # to help with visualization of clusters we sample points from each cluster (50 points per cluster in this case)
        sampled_indices=[]
        for cluster_id in range(k):
            cluster_indices=np.where(clusters==cluster_id)[0]
            if len(cluster_indices)>samples_per_cluster:
                # randomly sample indices from this cluster
                sampled_cluster_indices=np.random.choice(cluster_indices,size=samples_per_cluster,replace=False)
                sampled_indices.extend(sampled_cluster_indices)
            else:
                # if the cluster has fewer points than requested, take all of them
                sampled_indices.extend(cluster_indices)
        
        sampled_indices=np.array(sampled_indices) # convert to array for indexing
        
        # 3D scatter plot
        plt.figure(figsize=(10, 8))
        ax=plt.axes(projection='3d')
    
        sampled_X=X_pca[sampled_indices]
        sampled_clusters=clusters[sampled_indices]
        scatter=ax.scatter(sampled_X[:,0],sampled_X[:,1],sampled_X[:,2],c=sampled_clusters, cmap='tab20', alpha=0.7,edgecolors='k',s=50)
        
        # we also plot centroids
        ax.scatter(centroids[:,0],centroids[:,1],centroids[:,2],s=250,c='red',marker='X',label='Centroids',edgecolors='black')
        
        # add legend for clusters
        legend1=ax.legend(*scatter.legend_elements(),title="Clusters",loc="upper right")
        ax.add_artist(legend1)
        
        # add a separate legend for centroids
        ax.legend(loc="upper left")

        ax.set_title(f'K-Means Clustering (k={k})\n{samples_per_cluster} points shown per cluster', fontsize=12)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        
        ax.view_init(elev=30, azim=45) # set perspective for better 3D visualization
        
        #save plot
        plot_filename=f"kmeans_{k}_clusters.png"
        plt.tight_layout()
        plt.savefig(f"{output_path}/{plot_filename}", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"KMeans Clusters = {k} Plot saved to {output_path}/{plot_filename}")
    
    # visualization showing the distribution of original categories in each cluster
    for k in top_3_k:
        kmeans=KMeans(n_clusters=k, random_state=17, n_init=10)
        clusters=kmeans.fit_predict(X_pca)
        
        plt.figure(figsize=(14, 10))
        
        # create a crosstab between clusters and categories
        cluster_category_df=pd.DataFrame({'Cluster': clusters,'Category': categories_array})
        
        # plot a heatmap showing the distribution
        cross_tab=pd.crosstab(cluster_category_df['Cluster'],cluster_category_df['Category'],normalize='index')
        
        sns.heatmap(cross_tab, cmap='viridis', annot=True, fmt='.2f', linewidths=.5)
        plt.title(f'Distribution of Categories in Each Cluster (k={k})')

        #save plot
        plt.tight_layout()
        plot_filename=f"category_distribution_k{k}.png"
        plt.savefig(f"{output_path}/{plot_filename}", dpi=300, bbox_inches="tight")
        
        plt.close()
        print(f"Distribution of categories for clusters = {k} plot saved to {output_path}/{plot_filename}")

# def perform_hierarchical_clustering():
#     """
#     Performs hierarchical clustering on the PCA-reduced dataset and creates a dendrogram visualization.
#     """
#     # Get the PCA data
#     df_pca = get_3d_pca_data()
#     categories_array = df_pca["category"].values
#     df_features = df_pca.drop("category", axis=1)
    
#     # Create output directory
#     output_path = "../website/plots/hierarchical_clustering"
#     os.makedirs(output_path, exist_ok=True)
    
#     # Perform hierarchical clustering
#     # Using Ward's method as it minimizes the variance within clusters (similar to k-means objective)
#     Z = linkage(df_features, method='ward')
    
#     # Plot the dendrogram
#     plt.figure(figsize=(16, 10))
#     plt.title('Hierarchical Clustering Dendrogram', fontsize=16)
#     plt.xlabel('Sample index', fontsize=12)
#     plt.ylabel('Distance', fontsize=12)
    
#     # Generate dendrogram with colors for different clusters
#     # Truncate the dendrogram to show only the most important merges
#     dendrogram(Z, truncate_mode='lastp', p=30, leaf_font_size=10, 
#                show_contracted=True, leaf_rotation=90)
    
#     plot_filename="hierarchical_dendrogram.png"
#     plt.tight_layout()
#     plt.savefig(f"{output_path}/{plot_filename}", dpi=300, bbox_inches="tight")
#     plt.close()
#     print(f"Hierarchial dendogram plot saved to {output_path}/{plot_filename}")
    
#     # For the top 3 k values identified in the KMeans clustering
#     # Let's create cluster assignments and compare them with original categories
#     df_pca = get_3d_pca_data()
#     X_pca = df_pca.drop("category", axis=1).values
    
#     # Get the same top 3 k values from KMeans
#     silhouette_scores = []
#     K_range = range(2, 6)
#     for k in K_range:
#         kmeans = KMeans(n_clusters=k, n_init=10, random_state=17)
#         labels = kmeans.fit_predict(X_pca)
#         score = silhouette_score(X_pca, labels)
#         silhouette_scores.append((score, k))
    
#     top_3_k = [k for _, k in sorted(silhouette_scores, reverse=True)[:3]]
    
#     # For each of the top 3 cluster numbers, create and visualize hierarchical clusters
#     for k in top_3_k:
#         # Cut the dendrogram to get k clusters
#         hierarchical_clusters = fcluster(Z, k, criterion='maxclust')
        
#         # Create a dataframe with category and cluster assignments
#         cluster_category_df = pd.DataFrame({
#             'Hierarchical Cluster': hierarchical_clusters,
#             'Category': categories_array
#         })
        
#         # Plot distribution of categories in each hierarchical cluster
#         plt.figure(figsize=(14, 10))
#         cross_tab = pd.crosstab(
#             cluster_category_df['Hierarchical Cluster'],
#             cluster_category_df['Category'],
#             normalize='index'
#         )
        
#         plot_filename=f"hierarchical_category_distribution_k={k}.png"
#         sns.heatmap(cross_tab, cmap='viridis', annot=True, fmt='.2f', linewidths=.5)
#         plt.title(f'Distribution of Categories in Each Hierarchical Cluster (k={k})')
#         plt.tight_layout()
#         plt.savefig(f"{output_path}/{plot_filename}", 
#                     dpi=300, bbox_inches="tight")
#         plt.close()
#         print(f"Distribution of categories for clusters = {k} plot (Hierarchial) saved to {output_path}/{plot_filename}")

#         # 3D scatter plot for visualization (with limited samples for clarity)
#         samples_per_cluster = 50
#         sampled_indices = []
        
#         for cluster_id in range(1, k+1):  # Hierarchical cluster IDs start from 1
#             cluster_indices = np.where(hierarchical_clusters == cluster_id)[0]
#             if len(cluster_indices) > samples_per_cluster:
#                 sampled_cluster_indices = np.random.choice(
#                     cluster_indices, size=samples_per_cluster, replace=False
#                 )
#                 sampled_indices.extend(sampled_cluster_indices)
#             else:
#                 sampled_indices.extend(cluster_indices)
        
#         sampled_indices = np.array(sampled_indices)
        
#         # Create 3D scatter plot
#         plt.figure(figsize=(10, 8))
#         ax = plt.axes(projection='3d')
        
#         sampled_X = X_pca[sampled_indices]
#         sampled_clusters = hierarchical_clusters[sampled_indices]
        
#         scatter = ax.scatter(
#             sampled_X[:, 0], sampled_X[:, 1], sampled_X[:, 2],
#             c=sampled_clusters, cmap='tab20', alpha=0.7, edgecolors='k', s=50
#         )
        
#         # Calculate "centroids" for hierarchical clusters
#         centroids = []
#         for i in range(1, k+1):
#             mask = hierarchical_clusters == i
#             if np.any(mask):
#                 centroids.append(np.mean(X_pca[mask], axis=0))
        
#         centroids = np.array(centroids)
        
#         # Plot centroids
#         ax.scatter(
#             centroids[:, 0], centroids[:, 1], centroids[:, 2],
#             s=250, c='red', marker='X', label='Cluster Centers', edgecolors='black'
#         )
        
#         # Add legend for clusters
#         legend1 = ax.legend(*scatter.legend_elements(), title="Clusters", loc="upper right")
#         ax.add_artist(legend1)
        
#         # Add a separate legend for centroids
#         ax.legend(loc="upper left")
        
#         ax.set_title(f'Hierarchical Clustering (k={k})\n{samples_per_cluster} points shown per cluster', 
#                      fontsize=12)
#         ax.set_xlabel('PC1')
#         ax.set_ylabel('PC2')
#         ax.set_zlabel('PC3')
        
#         ax.view_init(elev=30, azim=45)
        
#         plot_filename=f"hierarchical_{k}_clusters.png"
#         plt.tight_layout()
#         plt.savefig(f"{output_path}/{plot_filename}", 
#                     dpi=300, bbox_inches="tight")
#         plt.close()
#         print(f"Hierarchial Clusters = {k} Plot saved to {output_path}/{plot_filename}")



def perform_hierarchical_clustering():
    """
    Performs hierarchical clustering on the PCA-reduced dataset using cosine similarity 
    and creates a dendrogram visualization.
    """
    # Get the PCA data
    df_pca = get_3d_pca_data()
    categories_array = df_pca["category"].values
    df_features = df_pca.drop("category", axis=1)
    
    # Create output directory
    output_path = "../website/plots/hierarchical_clustering"
    os.makedirs(output_path, exist_ok=True)
    
    # Compute pairwise cosine distances (1 - cosine similarity)
    cosine_distances = pdist(df_features, metric='cosine')
    
    # Perform hierarchical clustering using the computed cosine distances
    Z = linkage(cosine_distances, method='average')  # Using average linkage (UPGMA)

    # Plot the dendrogram
    plt.figure(figsize=(16, 10))
    plt.title('Hierarchical Clustering Dendrogram (Cosine Similarity)', fontsize=16)
    plt.xlabel('Sample index', fontsize=12)
    plt.ylabel('Distance', fontsize=12)

    # Generate dendrogram with colors for different clusters
    dendrogram(Z, truncate_mode='lastp', p=30, leaf_font_size=10, 
               show_contracted=True, leaf_rotation=90)

    plot_filename = "hierarchical_dendrogram.png"
    plt.tight_layout()
    plt.savefig(f"{output_path}/{plot_filename}", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Hierarchical dendrogram plot saved to {output_path}/{plot_filename}")
    
    # Identify the top 3 k values from KMeans clustering
    df_pca = get_3d_pca_data()
    X_pca = df_pca.drop("category", axis=1).values

    silhouette_scores = []
    K_range = range(2, 6)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=17)
        labels = kmeans.fit_predict(X_pca)
        score = silhouette_score(X_pca, labels)
        silhouette_scores.append((score, k))
    
    top_3_k = [k for _, k in sorted(silhouette_scores, reverse=True)[:3]]

    # Create hierarchical clusters for the top 3 k values
    for k in top_3_k:
        hierarchical_clusters = fcluster(Z, k, criterion='maxclust')

        # Create a dataframe with category and cluster assignments
        cluster_category_df = pd.DataFrame({
            'Hierarchical Cluster': hierarchical_clusters,
            'Category': categories_array
        })

        # Plot distribution of categories in each hierarchical cluster
        plt.figure(figsize=(14, 10))
        cross_tab = pd.crosstab(
            cluster_category_df['Hierarchical Cluster'],
            cluster_category_df['Category'],
            normalize='index'
        )

        plot_filename = f"hierarchical_category_distribution_k={k}.png"
        sns.heatmap(cross_tab, cmap='viridis', annot=True, fmt='.2f', linewidths=.5)
        plt.title(f'Distribution of Categories in Each Hierarchical Cluster (k={k})')
        plt.tight_layout()
        plt.savefig(f"{output_path}/{plot_filename}", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Distribution of categories for clusters = {k} plot (Hierarchical) saved to {output_path}/{plot_filename}")

        # 3D scatter plot visualization (with limited samples for clarity)
        samples_per_cluster = 50
        sampled_indices = []

        for cluster_id in range(1, k+1):
            cluster_indices = np.where(hierarchical_clusters == cluster_id)[0]
            if len(cluster_indices) > samples_per_cluster:
                sampled_cluster_indices = np.random.choice(
                    cluster_indices, size=samples_per_cluster, replace=False
                )
                sampled_indices.extend(sampled_cluster_indices)
            else:
                sampled_indices.extend(cluster_indices)

        sampled_indices = np.array(sampled_indices)

        # Create 3D scatter plot
        plt.figure(figsize=(10, 8))
        ax = plt.axes(projection='3d')

        sampled_X = X_pca[sampled_indices]
        sampled_clusters = hierarchical_clusters[sampled_indices]

        scatter = ax.scatter(
            sampled_X[:, 0], sampled_X[:, 1], sampled_X[:, 2],
            c=sampled_clusters, cmap='tab20', alpha=0.7, edgecolors='k', s=50
        )

        # Compute "centroids" for hierarchical clusters
        centroids = []
        for i in range(1, k+1):
            mask = hierarchical_clusters == i
            if np.any(mask):
                centroids.append(np.mean(X_pca[mask], axis=0))

        centroids = np.array(centroids)

        # Plot centroids
        ax.scatter(
            centroids[:, 0], centroids[:, 1], centroids[:, 2],
            s=250, c='red', marker='X', label='Cluster Centers', edgecolors='black'
        )

        # Add legend for clusters
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters", loc="upper right")
        ax.add_artist(legend1)

        # Add a separate legend for centroids
        ax.legend(loc="upper left")

        ax.set_title(f'Hierarchical Clustering (k={k})\n{samples_per_cluster} points shown per cluster', fontsize=12)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')

        ax.view_init(elev=30, azim=45)

        plot_filename = f"hierarchical_{k}_clusters.png"
        plt.tight_layout()
        plt.savefig(f"{output_path}/{plot_filename}", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Hierarchical Clusters = {k} Plot saved to {output_path}/{plot_filename}")



def perform_dbscan():
    """
    Performs DBSCAN clustering on the PCA-reduced dataset and creates visualizations.
    Automatically selects optimal epsilon and min_samples by scanning a range.
    """
    df_pca = get_3d_pca_data()
    categories_array = df_pca["category"].values
    df_features = df_pca.drop("category", axis=1)
    X_pca = df_features.values
    
    output_path = "../website/plots/dbscan"
    os.makedirs(output_path, exist_ok=True)
    
    # we find optimal epsilon using k-distance method
    k = int(np.sqrt(len(X_pca)))  # rule of thumb for k-distance graph
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X_pca)
    distances, _ = neigh.kneighbors(X_pca)
    
    # we find the optimal epsilon using KneeLocator
    sorted_distances = np.sort(distances[:, k - 1])
    kneedle = KneeLocator(range(len(sorted_distances)), sorted_distances, curve="convex", direction="increasing")
    epsilon_mid = sorted_distances[kneedle.knee] if kneedle.knee else np.percentile(sorted_distances, 95)
    
   
    epsilon_values = [round(epsilon_mid * factor, 3) for factor in [0.8, 1.0, 1.2]] # explore a range of epsilon values
    
    min_samples_values = [int(np.log(len(X_pca))) + i for i in range(-2, 3)] # select multiple min_samples values dynamically
    
    best_config = None
    max_clusters = 0
    
    # try different combinations of epsilon and min_samples
    for epsilon in epsilon_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
            clusters = dbscan.fit_predict(X_pca)
            
            # count meaningful clusters (exclude noise)
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            noise_ratio = np.sum(clusters == -1) / len(X_pca)
            
            if n_clusters > max_clusters and noise_ratio < 0.5:
                max_clusters = n_clusters
                best_config = (epsilon, min_samples, clusters)
    
    
    if best_config: #use the best configuration
        epsilon_optimal, min_samples_optimal, clusters = best_config
    else:
        epsilon_optimal, min_samples_optimal = epsilon_values[1], min_samples_values[2]
        dbscan = DBSCAN(eps=epsilon_optimal, min_samples=min_samples_optimal)
        clusters = dbscan.fit_predict(X_pca)
    
    print(f"Final DBSCAN Parameters: epsilon={epsilon_optimal}, min_samples={min_samples_optimal}")
    
    # compute statistics
    noise_points = np.sum(clusters == -1)
    non_noise_ratio = 1.0 - (noise_points / len(X_pca))
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    
    print(f"DBSCAN Results: {n_clusters} clusters, {non_noise_ratio:.2%} non-noise points")
    
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='3d')
    scatter = ax.scatter(
        X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
        c=clusters, cmap='tab20', alpha=0.7, edgecolors='k', s=50
    )
    
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters", loc="upper right")
    ax.add_artist(legend1)
    
    ax.set_title(f'DBSCAN Clustering (eps={epsilon_optimal}, min_samples={min_samples_optimal})', fontsize=12)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.view_init(elev=30, azim=45)
    
    plot_filename="dbscan_results.png"
    plt.tight_layout()
    plt.savefig(f"{output_path}/{plot_filename}.png", dpi=300, bbox_inches="tight")
    print(f"DBSCAN Clustering Plot saved to {output_path}/{plot_filename}")
    plt.close()
    
    # Generate category distribution heatmap
    cluster_category_df = pd.DataFrame({
        'DBSCAN Cluster': clusters,
        'Category': categories_array
    })
    
    plt.figure(figsize=(14, 10))
    cross_tab = pd.crosstab(
        cluster_category_df['DBSCAN Cluster'],
        cluster_category_df['Category'],
        normalize='index'
    )
    
    sns.heatmap(cross_tab, cmap='viridis', annot=True, fmt='.2f', linewidths=.5)
    plt.title(f'Distribution of Categories in DBSCAN Clusters')
    plt.tight_layout()
    plot_filename="dbscan_category_distribution.png"
    plt.savefig(f"{output_path}/{plot_filename}", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"DBSCAN Clustering Plot saved to {output_path}/{plot_filename}")

if __name__=="__main__":
    
    # perform_kmeans()
    perform_hierarchical_clustering()
    # perform_dbscan()