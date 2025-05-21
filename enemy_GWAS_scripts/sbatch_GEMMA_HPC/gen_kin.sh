#!/bin/bash
#SBATCH --job-name=gemma_kinship
#SBATCH --output=logs/gemma_kinship.out
#SBATCH --error=logs/gemma_kinship.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8

# Why use phenotype file, ensures the rows in the kinship matrix correspond exactly to the phenotype data in later testing
# Checks alignmnet

module load gemma/0.98.4

# Run GEMMA kinship matrix computation
gemma \
  -g genotypes/genotypes_305 \
  -p dat_tot_forGemma_Enemies.csv \
  -gk 1 \
  -o kinship_enemy_305














