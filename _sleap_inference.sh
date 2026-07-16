#! /bin/bash

#SBATCH -J sleap_inference
#SBATCH -p all

#SBATCH -c 7
#SBATCH --mem=250GB
#SBATCH --gpus=1
#SBATCH -t 12:00:00
#SBATCH --partition=witten

module load anacondapy/2023.07-cuda
eval "$(conda shell.bash hook)"
# conda activate sleap # NOTE: this is to use the old sleap instal on the cluster (doesn't work with the new yaml files).
conda activate sleap-update

# Print job, TF, and GPU information.
echo "Current conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"

# Variables passed as command line arguments.
video=$1
centroid_model=$2
centered_model=$3
tracks_file_path=$4

# Make file name the mouse color + _inference.h5.
output_filename="${mouse_color}_inference.h5"

# Run sleap inference.
sleap-track -o $tracks_file_path --tracking.tracker simple --batch_size 16 --peak_threshold 0.3 --tracking.track_window 3 --tracking.post_connect_single_breaks 1 --tracking.similarity instance --tracking.clean_instance_count 1 --model $centroid_model --model  $centered_model $video

# Convert slp file to h5 (for anipose).
h5_file_path="${tracks_file_path%.slp}.h5"
echo $tracks_file_path
sleap-convert --format analysis "${tracks_file_path}.slp"

# Check if the sleap command was successful.
if [ $? -eq 0 ]; then
    echo "Sleap inference completed successfully."
else
    echo "Sleap inference failed."
fi

echo "~~~All done!~~~"