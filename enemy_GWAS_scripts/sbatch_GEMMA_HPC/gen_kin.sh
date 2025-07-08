#!/bin/bash
#SBATCH --job-name=gemma_kinship
#SBATCH --output=logs/gemma_kinship.out
#SBATCH --error=logs/gemma_kinship.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G

module purge
module load gemma/0.98.4

gemma -bfile genotypes/genotypes_305 \
      -p enemies_forGemma/blups_SC_imputed_k7_forGemma.csv \
      -miss 0.1 \
      -gk 2 \
      -o kinship_enemy_305_impk7_miss0.1














