#! /bin/bash

#SBATCH -J anipose_triangulation
#SBATCH -p all

#SBATCH -c 1
#SBATCH --mem=100GB
#SBATCH -t 02:00:00

module load anacondapy/2023.07-cuda
eval "$(conda shell.bash hook)"
conda activate sleap-anipose

# Verify the current conda environment
echo "Current conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"

session_directory=$1
calibration_file=$2
output_filename=$3

# Retry mechanism for slap-triangulate command
max_retries=100
retry_count=0
success=0

while [ $retry_count -lt $max_retries ]; do
    slap-triangulate --p2d $session_directory --calib $calibration_file --fname $output_filename --scale_smooth 1 --n_deriv_smooth 2 --reproj_loss l2 --reproj_error_threshold 10
    if [ $? -eq 0 ]; then
        success=1
        break
    else
        echo "slap-triangulate failed, retrying in 5 seconds..."
        sleep 5
        retry_count=$((retry_count + 1))
    fi
done

if [ $success -eq 0 ]; then
    echo "Error: slap-triangulate failed after $max_retries retries."
    exit 1
fi

# Retry mechanism for slap-reproject command
reprojection_filename="${output_filename%.*}_reprojected.h5"
retry_count=0
success=0

while [ $retry_count -lt $max_retries ]; do
    slap-reproject --p3d $output_filename --calib $calibration_file --fname $reprojection_filename
    if [ $? -eq 0 ]; then
        success=1
        break
    else
        echo "slap-reproject failed, retrying in 5 seconds..."
        sleep 5
        retry_count=$((retry_count + 1))
    fi
done

if [ $success -eq 0 ]; then
    echo "Error: slap-reproject failed after $max_retries retries."
    exit 1
fi

echo "~~~All done!~~~"