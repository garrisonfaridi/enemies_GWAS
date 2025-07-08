def cluster_phenotypes_by_pca_loadings(loadings, trait_names, method="hierarchical", n_clusters=5, cut_height=0.4, plot=True):
    """
    Clusters phenotypes based on their PCA loadings and saves cluster assignment to CSV.

    Parameters:
    - loadings: numpy array of shape (n_traits, n_pcs)
    - trait_names: list or Index of trait names (length must match n_traits)
    - method: "kmeans" or "hierarchical"
    - n_clusters: for KMeans
    - cut_height: used only for hierarchical clustering
    - plot: whether to generate dendrogram/heatmap

    Returns:
    - cluster_labels: Series of cluster labels
    - sorted_df: DataFrame of sorted PCA loadings by cluster
    - phenos_by_cluster: Dict of cluster ID → trait names
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
    from sklearn.cluster import KMeans

    # Ensure trait_names is a list
    trait_names = list(trait_names)

    # Create DataFrame of PCA loadings
    df = pd.DataFrame(loadings, index=trait_names, columns=[f"PC{i+1}" for i in range(loadings.shape[1])])

    # Clustering
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = model.fit_predict(df.values)
    elif method == "hierarchical":
        Z = linkage(df.values, method="ward")
        clusters = fcluster(Z, t=cut_height, criterion="distance")

        if plot:
            plt.figure(figsize=(10, 6))
            dendrogram(Z, labels=trait_names, leaf_rotation=90, color_threshold=cut_height)
            plt.axhline(cut_height, color='red', linestyle='--', label=f"Cut at {cut_height}")
            plt.title("Dendrogram of PCA Loadings Clustering")
            plt.legend()
            plt.tight_layout()
            plt.savefig("pca_loadings_dendrogram.png", dpi=300)
            plt.show()
    else:
        raise ValueError("Method must be 'kmeans' or 'hierarchical'")

    # Assign and sort
    df["Cluster"] = clusters
    sorted_df = df.sort_values("Cluster")

    # Group by cluster
    phenos_by_cluster = {
        c: sorted_df[sorted_df["Cluster"] == c].index.tolist()
        for c in sorted_df["Cluster"].unique()
    }

    # Print output with count
    for cluster_id, phenos in phenos_by_cluster.items():
        print(f"Cluster {cluster_id} ({len(phenos)} Traits): {phenos}")

    # Format output for CSV
    output_df = pd.DataFrame({
        "Cluster": [f"Cluster {c}" for c in sorted_df["Cluster"]],
        "Phenotype": sorted_df.index,
        "Column Number": [trait_names.index(name) for name in sorted_df.index]
    })

    # Construct filename
    if method == "hierarchical":
        filename = f"pca_loadings_dend({cut_height}).csv"
    else:
        filename = f"pca_loadings_kmeans({n_clusters}).csv"

    # Save to CSV
    output_df.to_csv(filename, index=False)
    print(f"\nCluster information saved to: {filename}")

    # Optional heatmap
    if plot:
        plt.figure(figsize=(8, 0.4 * len(df)))
        sns.heatmap(sorted_df.drop(columns="Cluster"), annot=True, cmap="vlag", cbar=True)
        plt.title("PCA Loadings Cluster Heatmap")
        plt.xlabel("Principal Components")
        plt.ylabel("Traits")
        plt.tight_layout()
        plt.show()

    return pd.Series(clusters, index=df.index), sorted_df, phenos_by_cluster
