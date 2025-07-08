#!/bin/bash
#SBATCH --job-name=gemma_enemies
#SBATCH --array=1-42         # Update this to the number of traits (columns)
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=logs/enemy_trait_%A_%a.out
#SBATCH --error=logs/enemy_trait_%A_%a.err

# Load GEMMA
module load gemma/0.98.4

# File paths
PHENO="enemies_forGemma/blups_SC_imputed_k7_forGemma.csv"                     
BED="genotypes/genotypes_305"                        # Prefix for .bed/.bim/.fam files
KINSHIP="output/kinship_enemy_305_impk7_miss0.1.cXX.txt"          # Kinship matrix
OUTDIR="gemma_outputs_individuals_KNN_imputed_k7_kin_miss0.1"

# Create output directory if it doesn't exist
mkdir -p $OUTDIR

# GEMMA uses 1-based indexing for the -n argument
TRAIT_NUM=$SLURM_ARRAY_TASK_ID

# Run GEMMA
gemma -bfile $BED \
      -k $KINSHIP \
      -p $PHENO \
      -n $TRAIT_NUM \
      -lmm 1 \
      -maf 0.03 \
      -miss 0.1 \
      -o enemy_trait_${TRAIT_NUM} \
      -outdir $OUTDIR
