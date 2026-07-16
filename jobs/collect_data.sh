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

# Verify Python and CARLA client
echo "Python: $(python --version)"
python -c "import carla; print('CARLA client imported successfully')"

# Set display variables for offscreen rendering
export DISPLAY=
export SDL_VIDEODRIVER=offscreen
export SDL_HINT_CUDA_DEVICE=0

# Start CARLA server in background
echo "Starting CARLA server..."
$CARLA_ROOT/CarlaUE4.sh \
    -RenderOffScreen \
    -quality-level=Low \
    -carla-port=2000 \
    -nosound \
    -dx11 &

CARLA_PID=$!
echo "CARLA PID: $CARLA_PID"

# Wait longer for CARLA to initialise on HPC
echo "Waiting 60 seconds for CARLA to initialise..."
sleep 60

# Check if CARLA process is still running
if kill -0 $CARLA_PID 2>/dev/null; then
    echo "CARLA server process is alive — attempting connection..."
else
    echo "CARLA process died — checking for crash log..."
    # Try to find crash output
    find $SCRATCH -name "crash*.log" 2>/dev/null
    find /tmp -name "*carla*" 2>/dev/null
    echo "Attempting restart with Vulkan..."
    $CARLA_ROOT/CarlaUE4.sh \
        -RenderOffScreen \
        -quality-level=Low \
        -carla-port=2000 \
        -nosound &
    CARLA_PID=$!
    echo "Restarted CARLA PID: $CARLA_PID"
    sleep 60
fi

# Test CARLA connection
echo "Testing CARLA connection..."
python -c "
import carla
import time
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(30.0)
    world = client.get_world()
    print(f'CARLA connected: {world.get_map().name}')
except Exception as e:
    print(f'Connection failed: {e}')
    exit(1)
"

# Run data collection
echo "Starting data collection..."
cd $CODE_ROOT/data_collection
python collect_episodes.py \
    --num_episodes 500 \
    --max_steps 500 \
    --save_dir /scratch/prj/inf_offroad_auto_nav/data/raw

# Run RTG computation
echo "Computing return-to-go..."
python compute_rtg.py \
    --raw_path /scratch/prj/inf_offroad_auto_nav/data/raw/episodes_final.pkl \
    --save_path /scratch/prj/inf_offroad_auto_nav/data/processed/dataset.pkl

# Kill CARLA
kill $CARLA_PID 2>/dev/null
echo "Job finished: $(date)"