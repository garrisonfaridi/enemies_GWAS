"""
Optimized KNN Imputation with MSE Tracking using 10 fold masking of NAs removed dataset
"""
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Read scaled centered BLUPS
blups_SC = pd.read_csv("forFabrice_allBLUPs_scaledCentered_2023-08-17.csv")

# Remove NA rows from blups_SC
blups_SC_noNA = blups_SC.dropna()
print(blups_SC_noNA.shape)

# Remove metadata columns from blups_SC_noNA (first 4 columns)
blups_SC_noNA = blups_SC_noNA.iloc[:, 4:]

# Copy the original data for later comparison
original = blups_SC_noNA.copy()

# Set parameters
# Number of neighbors to test
k_values = range(1, 21)

# Number of replicates
replicates = 10

# Number of rows in the original dataset
n_rows = original.shape[0]

# To store MSE values for each replicate
mse_results_avg = {k: [] for k in k_values}

# Begin replicates
for rep in range(replicates):
    masked = original.copy()
    mask = pd.DataFrame(False, index=masked.index, columns=masked.columns)

    # Randomly mask 5% of values in each column
    for col in masked.columns:
        n_mask = int(0.05 * n_rows)
        mask_indices = np.random.choice(n_rows, n_mask, replace=False)
        masked.iloc[mask_indices, masked.columns.get_loc(col)] = np.nan
        mask.iloc[mask_indices, mask.columns.get_loc(col)] = True

    # Evaluate MSE for each k
    for k in k_values:
        imputer = KNNImputer(n_neighbors=k)
        imputed = imputer.fit_transform(masked)
        imputed_df = pd.DataFrame(imputed, columns=masked.columns, index=masked.index)

        mse_per_col = []
        for col in masked.columns:
            true_vals = original.loc[mask[col], col]
            pred_vals = imputed_df.loc[mask[col], col]
            mse = mean_squared_error(true_vals, pred_vals)
            mse_per_col.append(mse)

        mse_results_avg[k].append(np.mean(mse_per_col))

# Average across replicates
avg_mse_per_k = {k: np.mean(mse_results_avg[k]) for k in k_values}

# Find k with the lowest average MSE
best_k = min(avg_mse_per_k, key=avg_mse_per_k.get)
best_mse = avg_mse_per_k[best_k]
print(f"Best k: {best_k} with MSE: {best_mse:.4f}")

# Elbow Plot
plt.figure(figsize=(8, 5))
plt.plot(list(avg_mse_per_k.keys()), list(avg_mse_per_k.values()), marker='o')
plt.title("KNN Imputation Elbow Plot (10 Replicates)")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Average MSE")
plt.xticks(k_values)
plt.grid(True)
plt.tight_layout()
plt.show()

# Impute each of these datasets using KNN imputation
# Initialize imputer
imputer = KNNImputer(n_neighbors=best_k)

# Isolate the numeric portion of the dataset
blups_SC_meta = blups_SC.iloc[:, :4]
blups_SC_numeric = blups_SC.iloc[:, 4:]

# Impute only the numeric portion

blups_SC_numeric_imputed = pd.DataFrame(imputer.fit_transform(blups_SC_numeric),
                                        columns=blups_SC_numeric.columns)

# Combine metadata and imputed values
blups_SC_imputed = pd.concat([blups_SC_meta, blups_SC_numeric_imputed], axis=1)

# View the header of each dataset
print(blups_SC_imputed.head())

# Save the imputed datasets
blups_SC_imputed.to_csv("allBLUPs_scaledCentered_imputed_k7.csv", index=False)