#!/bin/bash
#SBATCH --job-name=fast_sweep
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=32G

module load gcc/11.4.0
module load openmpi/4.1.4
module load python/3.11.4
source ~/envs/smallformer/bin/activate

python scripts/hparam_sweep.py
