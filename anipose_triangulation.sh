#! /bin/bash

#SBATCH -J anipose_triangulation
#SBATCH -p all

#SBATCH -c 1
#SBATCH --mem=8GB
#SBATCH -t 01:00:00

#SBATCH --error=SBATCH_outputs/anipose_triangulation_error.txt
#SBATCH --output=SBATCH_outputs/anipose_triangulation_output.txt

module load anacondapy/2023.07-cuda
source activate sleap-anipose

session_directory=$1
calibration_file=$2
output_filename=$3

# Run triangulation.
slap-triangulate --p2d $session_directory --calib $calibration_file --fname $output_filename

# Reproject to each view.
reprojection_filename="${output_filename%.*}_reprojected.h5"
slap-reproject --p3d $output_filename --calib $calibration_file --fname reprojection_filename

echo "~~~All done!~~~"