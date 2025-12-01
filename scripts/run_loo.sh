#!/bin/bash
#SBATCH --job-name=dickens_loo
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --mem=32G

# Required module dependencies for python 3.11.4
module load gcc/11.4.0
module load openmpi/4.1.4
module load python/3.11.4

# Activate your environment
source ~/envs/smallformer/bin/activate

# Run LOO training
python scripts/loo_train.py \
    --tokens texts/tokenized/tokens.npy \
    --offsets texts/tokenized/doc_offsets.npy \
    --tokenizer tokenizer.pt \
    --block_size 32 \
    --batch_size 32 \
    --epochs 3 \
    --lr 7e-4 \
    --d_model 384 \
    --n_layers 2 \
    --n_heads 2 \
    --d_ff 1024 \
    --dropout 0.1 \
    --downsample 100
