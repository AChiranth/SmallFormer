#!/bin/bash
#SBATCH --job-name=dickens_loo
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G

module load python/3.10
source ~/envs/smallformer/bin/activate

python scripts/loo_evaluate.py