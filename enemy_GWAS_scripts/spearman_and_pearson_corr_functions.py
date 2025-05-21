def correlate_phenotypes_with_phate(df, phate_cols=["PHATE1", "PHATE2"], id_cols=["ID", "SAMPLE", "ecotype_id", "pop_code"]):
    """
    Computes the Pearson correlation between each phenotype and PHATE1/PHATE2 coordinates.

    Returns:
    - correlation_df: DataFrame where rows = phenotypes, columns = PHATE1/PHATE2 correlations
    """
    pheno_cols = df.columns.difference(phate_cols + id_cols)
    
    corrs = {}
    for pheno in pheno_cols:
        corrs[pheno] = {}
        for phate in phate_cols:
            corrs[pheno][phate] = df[pheno].corr(df[phate])  # Pearson correlation
    
    correlation_df = pd.DataFrame(corrs).T  # phenotypes x PHATE coords
    correlation_df.columns = [f"Corr_{c}" for c in correlation_df.columns]
    
    return correlation_df
from scipy.stats import spearmanr
import pandas as pd

def spearman_correlate_phenotypes_with_phate(df, phate_cols=["PHATE1", "PHATE2"], id_cols=["ID", "SAMPLE", "ecotype_id", "pop_code"]):
    """
    Computes the Spearman correlation between each phenotype and PHATE coordinates.

    Returns:
    - correlation_df: DataFrame where rows = phenotypes, columns = Spearman correlations with PHATE coordinates
    """
    pheno_cols = df.columns.difference(phate_cols + id_cols)
    
    corrs = {}
    for pheno in pheno_cols:
        corrs[pheno] = {}
        for phate in phate_cols:
            rho, _ = spearmanr(df[pheno], df[phate], nan_policy='omit')
            corrs[pheno][phate] = rho  # Spearman rho
    
    correlation_df = pd.DataFrame(corrs).T
    correlation_df.columns = [f"Spearman_{c}" for c in correlation_df.columns]
    
    return correlation_df