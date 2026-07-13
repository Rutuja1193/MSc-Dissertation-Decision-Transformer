#!/bin/bash
#SBATCH --job-name=carla_collect
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/prj/inf_offroad_auto_nav/logs/collect_%j.out
#SBATCH --error=/scratch/prj/inf_offroad_auto_nav/logs/collect_%j.err

echo "Job started: $(date)"
echo "Running on node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Load modules
module load anaconda3/2022.10-gcc-13.2.0
source activate dt_offroad

# Paths
SCRATCH=/scratch/prj/inf_offroad_auto_nav
CARLA_ROOT=$SCRATCH/carla
CODE_ROOT=$SCRATCH/code

# Add CARLA Python API to path
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.16-py3.10-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla

# Start CARLA server in background
echo "Starting CARLA server..."
$CARLA_ROOT/CarlaUE4.sh \
    -RenderOffScreen \
    -quality-level=Low \
    -carla-port=2000 \
    -nosound &

CARLA_PID=$!
echo "CARLA PID: $CARLA_PID"

# Wait for CARLA to initialise
echo "Waiting 30 seconds for CARLA to initialise..."
sleep 30

# Check CARLA is running
if kill -0 $CARLA_PID 2>/dev/null; then
    echo "CARLA server is running"
else
    echo "ERROR: CARLA server failed to start"
    exit 1
fi

# Run data collection
echo "Starting data collection..."
cd $CODE_ROOT/data_collection
python collect_episodes.py

# Run RTG computation and dataset preparation
echo "Computing return-to-go and preparing dataset..."
python compute_rtg.py

# Kill CARLA server
kill $CARLA_PID
echo "CARLA server stopped"

echo "Job finished: $(date)"