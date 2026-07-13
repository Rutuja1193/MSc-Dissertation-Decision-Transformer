#!/bin/bash
#SBATCH --job-name=train_dt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/prj/inf_offroad_auto_nav/logs/train_dt_%j.out
#SBATCH --error=/scratch/prj/inf_offroad_auto_nav/logs/train_dt_%j.err

echo "Job started: $(date)"
echo "Running on node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Load modules
module load anaconda3/2022.10-gcc-13.2.0
source activate dt_offroad

# Paths
SCRATCH=/scratch/prj/inf_offroad_auto_nav
CODE_ROOT=$SCRATCH/code

# W&B key set via environment variable in ~/.bashrc
echo "W&B API key: $WANDB_API_KEY" | cut -c1-20

# Run training for 3 seeds
echo "Training Decision Transformer..."
for SEED in 0 1 2; do
    echo "Training seed $SEED..."
    python $CODE_ROOT/models/train_dt.py --seed $SEED
done

echo "Job finished: $(date)"