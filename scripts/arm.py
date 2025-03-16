import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder
import networkx as nx
import seaborn as sns
import os
import matplotlib.cm as cm

def make_transaction_data():
    """
    Function to create transaction data from audio features.

    This function:
    - Reads feature CSVs for different audio categories and concatenates them.
    - Samples 100 audio instances per category for balanced representation.
    - Selects a subset of audio features relevant for analysis.
    - Converts numerical feature values into categorical bins (Low, Medium, High).
    - Converts each row into a transaction (list of categorized feature values).
    - Saves the transactions to a CSV file for association rule mining.

    Parameters:
    - None

    Returns:
    - None
    """
    with open("../scripts/categories.txt","r") as f:
        categories=f.read().split()

    dfs=[pd.read_csv(f"../audiofiles_processed_features_CSVs/{category}_features.csv") for category in categories]
    df=pd.concat(dfs,ignore_index=True)

    # we take 100 samples per category
    df_sampled=df.groupby("category").sample(n=100,random_state=17).reset_index(drop=True)

    # select a subset of audio features
    selected_features=['mfcc_1','mfcc_2','mfcc_4','chroma_3','chroma_7','spectral_contrast_1','spectral_contrast_3','zero_crossing_rate',
                         'spectral_centroid','spectral_bandwidth','rms_energy','spectral_rolloff','tonnetz_1','tonnetz_3','tonnetz_6']

    df_filtered=df_sampled[["category"] + selected_features].copy()

    # make values categorical and handle dtype explicitly
    # less than 33% is marked as low 
    # above 33% is marked as high
    # in between is marked as medium
    for col in selected_features:
        # convert column to string dtype to avoid FutureWarning
        df_filtered[col]=df_filtered[col].astype('string')
        low_threshold=df_filtered[col].astype(float).quantile(0.33)  # convert back to float for quantile
        high_threshold=df_filtered[col].astype(float).quantile(0.66)
        df_filtered.loc[:,col]=df_filtered[col].apply(
            lambda x: f"{col}_Low" if float(x) <= low_threshold  
            else (f"{col}_Medium" if float(x) <= high_threshold 
                  else f"{col}_High")
        )

    # convert each row into a transaction (list of items)
    df_filtered.loc[:,"transaction"]=df_filtered.apply(lambda row: list(row),axis=1) # assign transaction column using .loc to avoid SettingWithCopyWarning
    transactions=df_filtered["transaction"].tolist() # extract these transactions

    # save transactions as csv
    output_path="../audio_features_transaction_form"
    os.makedirs(output_path,exist_ok=True)
    filename="transactions.csv"
    with open(f"{output_path}/{filename}","w") as f:
        for transaction in transactions:
            f.write(",".join(transaction) + "\n")
    print(f"Transactions saved to {output_path}/{filename}")

def perform_arm():
    """
    Function to perform Association Rule Mining (ARM) using the Apriori algorithm.

    This function:
    - Loads transaction data from the preprocessed CSV file.
    - Encodes transaction data into a binary format using TransactionEncoder.
    - Runs the Apriori algorithm to find frequent itemsets with a minimum support threshold.
    - Extracts association rules based on confidence and lift thresholds.
    - Formats the rules for readability and selects the top 15 rules based on support, confidence, and lift.
    - Saves the top rules into CSV files for further analysis.
    - Calls `plot_association_network` to visualize the top rules using a network graph.

    Parameters:
    - None

    Returns:
    - None
    """
    
    # get the transactions data
    transactions=[]
    output_path="../audio_features_transaction_form"
    filename="transactions.csv"
    with open(f"{output_path}/{filename}", "r") as f:
        transactions=[line.strip().split(",") for line in f.readlines()]

    # encode the transaction data to perform apriori
    te=TransactionEncoder()
    te_array=te.fit(transactions).transform(transactions)
    df_encoded=pd.DataFrame(te_array, columns=te.columns_)

    min_support=0.1 #we use minimum support of 0.1 
    min_confidence=0.7 #and minimum confidence of 0.7

    # run Apriori Algorithm with the defined thresholds
    frequent_itemsets=apriori(df_encoded, min_support=min_support, use_colnames=True)

    # extract Association Rules with filtered thresholds
    rules=association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    rules=rules[rules["antecedents"] != rules["consequents"]] # remove redundant rules where antecedents == consequents

    # format antecedents and consequents as readable strings
    rules["antecedents"]=rules["antecedents"].apply(lambda x: ", ".join(list(x)))
    rules["consequents"]=rules["consequents"].apply(lambda x: ", ".join(list(x)))

    # we keep only required columns
    rules_filtered=rules[["antecedents", "consequents", "support", "confidence", "lift", "leverage"]]

    # extract top 15 rules for support, confidence, and lift
    top_support=rules_filtered.sort_values(by="support", ascending=False).head(15)
    top_confidence=rules_filtered.sort_values(by="confidence", ascending=False).head(15)
    top_lift=rules_filtered.sort_values(by="lift", ascending=False).head(15)
    # save these to CSV files
    top_support.to_csv(f"{output_path}/top_15_support.csv", index=False)
    top_confidence.to_csv(f"{output_path}/top_15_confidence.csv", index=False)
    top_lift.to_csv(f"{output_path}/top_15_lift.csv", index=False)

    print(f"Performed association rule mining and saved top 15 rules for support, confidence and lift: {output_path}")

    plot_association_network(top_lift)


def plot_association_network(rules_df):
    """
    Function to visualize association rules as a directed network graph.

    This function:
    - Creates a directed graph using NetworkX, where nodes represent feature itemsets.
    - Adds edges between antecedents and consequents, weighted by the lift value.
    - Uses a force-directed layout for better readability.
    - Colors nodes based on their role (antecedents, consequents, or both).
    - Adjusts node sizes based on their degree in the graph.
    - Draws labels with background highlighting for clarity.
    - Adds a colorbar to indicate lift values.
    - Saves the network visualization as an image.

    Parameters:
    - rules_df (DataFrame): DataFrame containing association rules with antecedents, consequents, and lift values.

    Returns:
    - None
    """
    
    # create directed graph
    G=nx.DiGraph()
    
    rules=rules_df.copy()
    rules['antecedents']=rules['antecedents']
    rules['consequents']=rules['consequents']
    
    # add nodes and edges with proper attributes
    for _, row in rules.iterrows():
        antecedents=row['antecedents']
        consequents=row['consequents']
        lift=row['lift']
        
        # add nodes if they don't exist
        if not G.has_node(antecedents):
            G.add_node(antecedents, type='antecedent')
        if not G.has_node(consequents):
            G.add_node(consequents, type='consequent')
        
        G.add_edge(antecedents, consequents, weight=lift, width=lift/2) # add edge with lift as weight and width
    
    # create figure with explicit axes
    fig, ax=plt.subplots(figsize=(16, 14))    
    # spring layout with higher k value to spread nodes more
    pos=nx.spring_layout(G, k=1.5, seed=42)
    
    # get edge weights and normalize weights for coloring
    edge_weights=[G[u][v]['weight'] for u, v in G.edges()]
    min_weight=min(edge_weights)
    max_weight=max(edge_weights)
    norm=plt.Normalize(min_weight, max_weight)
    
    # set node colors based on type
    node_colors=[]
    for node in G.nodes():
        # check if node appears as both antecedent and consequent
        if any(node == row['antecedents'] for _, row in rules.iterrows()) and \
           any(node == row['consequents'] for _, row in rules.iterrows()):
            node_colors.append('purple')
        elif G.nodes[node]['type'] == 'antecedent':
            node_colors.append('lightblue')
        else:
            node_colors.append('lightgreen')
    
    # adjust node size based on degree
    node_sizes=[700 + (G.degree(node) * 100) for node in G.nodes()]
    
    # draw the network
    # draw edges with width proportional to lift and color mapped to lift value
    edges=nx.draw_networkx_edges(
        G, pos,
        width=[G[u][v]['width'] for u, v in G.edges()],
        edge_color=edge_weights,
        edge_cmap=cm.viridis,
        alpha=0.7,
        arrowsize=20,
        ax=ax
    )
    
    # draw nodes
    nodes=nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.8,
        edgecolors='black',
        ax=ax
    )
    
    # draw labels with shortened text and background
    # shorten labels for display
    def shorten_label(label, max_length=30):
        if len(label) > max_length:
            return label[:max_length] + "..."
        return label
    
    labels={node: shorten_label(node) for node in G.nodes()}
    
    # draw labels with white background
    bbox_props=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.7)
    nx.draw_networkx_labels(G, pos, labels, font_size=10, bbox=bbox_props, ax=ax)
    
    # add a colorbar for lift values
    sm=plt.cm.ScalarMappable(cmap=cm.viridis, norm=norm)
    sm.set_array([])
    cbar=fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Lift Value', rotation=270, labelpad=15)
    
    #add legend
    legend_elements=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                  markersize=15, label='Antecedent only'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', 
                  markersize=15, label='Consequent only'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', 
                  markersize=15, label='Both antecedent & consequent')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # set title and turn off axis
    ax.set_title("Audio Feature Association Rules Network", fontsize=16, fontweight="bold", pad=20)
    ax.axis('off')
    
    # add node count information
    rule_count=len(rules)
    node_count=len(G.nodes())
    ax.text(0.02, 0.02, f"Displaying {node_count} unique itemsets from {rule_count} rules", 
            transform=ax.transAxes, fontsize=12, 
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
    
    # save the plot
    output_path="../website/plots/arm"
    os.makedirs(output_path,exist_ok=True)
    filename="association_network.png"

    plt.tight_layout()
    plt.savefig(f"{output_path}/{filename}", dpi=300, bbox_inches="tight")
    print(f"Saved Network Visualization: {output_path}/{filename}")
    plt.close()

if __name__=="__main__":
    make_transaction_data()
    perform_arm()