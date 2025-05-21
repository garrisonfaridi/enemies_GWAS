"""
PHATE embedding of the scaled and centered BLUPs dataset
"""
import pandas as pd
import phate
import matplotlib.pyplot as plt

# Load the dataset
blups_SC_knn = pd.read_csv("allBLUPs_scaledCentered_imputed_k7.csv")

# Extract numeric features (exclude metadata)
X = blups_SC_knn.iloc[:, 4:]

print(X.shape)

# Run PHATE with optimized parameters
phate_operator = phate.PHATE(knn=5, t=100, n_components=2, n_jobs=-1, random_state=42)
X_phate = phate_operator.fit_transform(X)

# Store in a DataFrame
phate_df = pd.DataFrame(X_phate, columns=["PHATE1", "PHATE2"])
phate_df = pd.concat([blups_SC_knn.iloc[:, :4].reset_index(drop=True), phate_df], axis=1)

# Visualize
plt.figure(figsize=(8, 6))
plt.scatter(phate_df["PHATE1"], phate_df["PHATE2"], alpha=0.7)
plt.title("PHATE Embedding (knn=5, t=100)")
plt.xlabel("PHATE1")
plt.ylabel("PHATE2")
plt.grid(True)
plt.show()

# Save PHATE coordinates
phate_df.to_csv("phate_coords_SC_knn5_t100_impk7.csv", index=False)