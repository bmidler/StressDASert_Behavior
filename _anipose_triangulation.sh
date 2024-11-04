#! /bin/bash

#SBATCH -J anipose_triangulation
#SBATCH -p all

#SBATCH -c 1
#SBATCH --mem=32GB
#SBATCH -t 01:00:00

module load anacondapy/2023.07-cuda
eval "$(conda shell.bash hook)"
conda activate sleap-anipose

# Verify the current conda environment
echo "Current conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"

session_directory=$1
calibration_file=$2
output_filename=$3

# Run triangulation.
slap-triangulate --p2d $session_directory --calib $calibration_file --fname $output_filename --scale_smooth 1 --n_deriv_smooth 2

# Reproject to each view.
reprojection_filename="${output_filename%.*}_reprojected.h5"
slap-reproject --p3d $output_filename --calib $calibration_file --fname $reprojection_filename

echo "~~~All done!~~~"