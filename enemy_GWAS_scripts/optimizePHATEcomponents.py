"""
Optimize PHATE finding optimal numbero of components
"""
import pandas as pd
import phate
import matplotlib.pyplot as plt
import numpy as np

# Read the imputed dataset
blups_SC_knn = pd.read_csv("allBLUPs_scaledCentered_imputed_k7.csv")
X = blups_SC_knn.iloc[:, 4:]

# Run PHATE
phate_op = phate.PHATE()
X_phate = phate_op.fit_transform(X)
evals = phate_op.diff_potential

# Calculate mean eigenvalues across rows
mean_evals = np.mean(evals, axis=0)
log_mean_evals = np.log10(mean_evals)  # use log10 for plotting on log scale

# Fit a linear trendline in log space
x = np.arange(len(mean_evals))
z = np.polyfit(x, log_mean_evals, deg=1)
p = np.poly1d(z)

# Plot
plt.plot(x, mean_evals, label='Mean Eigenvalue')
plt.plot(x, 10**p(x), linestyle='--', label='Log Trendline')  # convert back to linear scale
plt.yscale('log')
plt.title("Average Diffusion Operator Eigenvalues with Trendline")
plt.xlabel("Component")
plt.ylabel("log Eigenvalue")
plt.legend()
plt.show()
