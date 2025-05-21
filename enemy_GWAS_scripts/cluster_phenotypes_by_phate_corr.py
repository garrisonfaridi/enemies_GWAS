import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import os

def cluster_phenotypes_by_phate_corr(correlation_df, method, n_clusters=5, cut_height=0.4, plot=True):
    """
    Clusters phenotypes based on correlation with PHATE coordinates.

    Parameters:
    - correlation_df: DataFrame with phenotype correlations
    - method: "kmeans" or "hierarchical"
    - n_clusters: used only for KMeans or AgglomerativeClustering
    - cut_height: dendrogram cut height (used only if method="hierarchical")
    - plot: whether to plot heatmap and dendrogram

    Returns:
    - cluster_labels: Series of cluster labels
    - sorted_corr_df: Correlation matrix sorted by cluster
    - phenos_by_cluster: Dictionary of cluster_id -> list of phenotype names
    """
    data = correlation_df.copy()

    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = model.fit_predict(data.values)

    elif method == "hierarchical":
        Z = linkage(data.values, method="ward")
        # Cut tree manually using distance threshold
        clusters = fcluster(Z, t=cut_height, criterion="distance")

        # Plot dendrogram with cut line
        if plot:
            plt.figure(figsize=(10, 6))
            dendrogram(
                Z,
                labels=data.index,
                leaf_rotation=90,
                color_threshold=cut_height,
                above_threshold_color='gray'
            )
            plt.axhline(y=cut_height, color='red', linestyle='--', label=f"Cut at {cut_height}")
            plt.title("Dendrogram of Phenotype Hierarchical Clustering")
            plt.xlabel("Phenotypes")
            plt.ylabel("Distance")
            plt.legend()
            plt.tight_layout()
            plt.show()
    else:
        raise ValueError("Method must be 'kmeans' or 'hierarchical'")

    # Assign cluster labels and sort data
    data["Cluster"] = clusters
    sorted_data = data.sort_values("Cluster")

    # Format for CSV output
    output_df = pd.DataFrame({
        "Cluster": ["Cluster {}".format(c) for c in sorted_data["Cluster"]],
        "Phenotype": sorted_data.index,
        "Column Number": [correlation_df.index.get_loc(name) for name in sorted_data.index]
    })

    # Save to CSV
    if method == "hierarchical":
        filename = f"corr_dend({cut_height}).csv"
    else:
        filename = f"corr_kmeans({n_clusters}).csv"

    output_df.to_csv(filename, index=False)
    print(f"\n Cluster information saved to: {filename}")

    # Create cluster → phenotype mapping
    phenos_by_cluster = {
        cluster_id: sorted_data[sorted_data["Cluster"] == cluster_id].index.tolist()
        for cluster_id in sorted_data["Cluster"].unique()
    }

    # Print clusters
    for cluster_id, phenos in phenos_by_cluster.items():
        print(f"Cluster {cluster_id}: {phenos}")

    # Heatmap
    if plot:
        plt.figure(figsize=(8, 0.4 * len(data)))
        sns.heatmap(sorted_data.drop("Cluster", axis=1), cmap="vlag", annot=True, cbar=True)
        plt.title(f"Phenotype Clustering by PHATE Correlation ({method})")
        plt.xlabel("PHATE Correlation")
        plt.ylabel("Phenotypes (Grouped)")
        plt.tight_layout()
        plt.show()

    return pd.Series(clusters, index=data.index), sorted_data, phenos_by_cluster
