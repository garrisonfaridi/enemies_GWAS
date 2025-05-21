#!/bin/bash
#SBATCH --job-name=gemma_array
#SBATCH --array=1-16          # Update if you have more clusters
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --output=logs/gemma_cluster_%A_%a.out
#SBATCH --error=logs/gemma_cluster_%A_%a.err

# Load GEMMA 
module load gemma/0.98.4

# Define paths
PHENO_DIR="impk7_knn5_t100_pears_0.35"
OUT_DIR="gemma_outputs_clusters_impk7_knn5_t100_pears_0.35"
KINSHIP="output/kinship_enemy_305.cXX.txt"
BED="genotypes/genotypes_305"

# Create output directory if needed
mkdir -p $OUT_DIR

# Run GEMMA on the phenotype file for this cluster
PHENO_FILE="${PHENO_DIR}/cluster_${SLURM_ARRAY_TASK_ID}_impk7_knn5_t100_pears_0.35.phenos.txt"

# Run GEMMA for MVLMM
gemma -lmm 1 \
      -maf 0.03 \
      -miss 0.1 \
      -bfile $BED \
      -k $KINSHIP \
      -p $PHENO_FILE \
      -maf 0.03 \
      -o cluster_${SLURM_ARRAY_TASK_ID} \
      -outdir $OUT_DIR
