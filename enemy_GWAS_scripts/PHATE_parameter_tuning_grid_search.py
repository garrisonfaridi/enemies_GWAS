import phate
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

def tune_phate_knn_t(X, knn_range=[5, 10, 20, 30], t_range=[5, 20, 50, 100], n_clusters=5):
    """
    Runs a grid search over PHATE's knn and t parameters and scores using silhouette score.
    
    Args:
        X (np.array or pd.DataFrame): Input feature matrix.
        knn_range (list): List of knn values to try.
        t_range (list): List of t values (diffusion time) to try.
        n_clusters (int): Number of clusters for KMeans.
    
    Returns:
        pd.DataFrame: Results with columns ['knn', 't', 'silhouette']
    """
    results = []
    
    for knn in tqdm(knn_range, desc="Tuning knn"):
        for t in t_range:
            try:
                phate_op = phate.PHATE(knn=knn, t=t, n_jobs=-1)
                X_phate = phate_op.fit_transform(X)
                labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(X_phate)
                score = silhouette_score(X_phate, labels)
                results.append({'knn': knn, 't': t, 'silhouette': score})
            except Exception as e:
                print(f"Failed for knn={knn}, t={t}: {e}")
                results.append({'knn': knn, 't': t, 'silhouette': np.nan})
    
    df = pd.DataFrame(results)
    return df
"""
Example usage:
X = blups_SC_knn.iloc[:, 4:]
df_results = tune_phate_knn_t(X, knn_range=[5,10,15], t_range=[10, 30, 50, 100], n_clusters=5)
best = df_results.sort_values('silhouette', ascending=False).head()
print(best)
"""

