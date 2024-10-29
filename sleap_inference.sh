#! /bin/bash

#SBATCH -J sleap_inference
#SBATCH -p all

#SBATCH -c 4
#SBATCH --mem=32GB
#SBATCH --gpus=1
#SBATCH -t 01:00:00

module load anacondapy/2023.07-cuda
source activate sleap

# Variables passed as command line arguments.
video=$1
centroid_model=$2
centered_model=$3
mouse_color=$4

# Run sleap inference.
sleap-track -o "${mouse_color}_inference" --tracking.tracker flow --peak_threshold 0.3 --tracking.track_window 3 --tracking.post_connect_single_breaks 1 --tracking.similarity instance --tracking.clean_instance_count 1 --model $centroid_model --model  $centered_model $video

# Check if the sleap command was successful.
if [ $? -eq 0 ]; then
    echo "Sleap inference completed successfully."
else
    echo "Sleap inference failed."
fi

echo "~~~All done!~~~"