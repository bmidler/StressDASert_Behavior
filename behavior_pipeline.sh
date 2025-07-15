#! /bin/bash

#SBATCH -J behavior_pipeline
#SBATCH -p all

#SBATCH -c 1
#SBATCH --mem=32GB
#SBATCH -t 12:00:00

#SBATCH --error=SBATCH_outputs/behavior_pipeline_error_%j_%x_%A_%a_%N_%t.txt
#SBATCH --output=SBATCH_outputs/behavior_pipeline_output_%j_%x_%A_%a_%N_%t.txt

# Check if day directory is provided as argument
if [ $# -eq 0 ]; then
    echo "Usage: sbatch behavior_pipeline.sh <day_directory>"
    exit 1
fi

DAY_DIRECTORY=$1

# Validate that the directory exists
if [ ! -d "$DAY_DIRECTORY" ]; then
    echo "Error: Directory $DAY_DIRECTORY does not exist"
    exit 1
fi

module load anacondapy/2023.07-cuda
source activate general
echo "Starting behavior pipeline for day directory: $DAY_DIRECTORY"
echo "Current conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"

# Find all session directories in the day directory
echo "Searching for session directories in: $DAY_DIRECTORY"

# Process each subdirectory as a session
for session_dir in "$DAY_DIRECTORY"/*; do
    if [ -d "$session_dir" ]; then
        session_name=$(basename "$session_dir")
        
        # Skip calibration directories
        if [ "$session_name" = "calibration" ]; then
            echo "Skipping calibration directory: $session_name"
            continue
        fi
        
        echo "Found session directory: $session_name"
        echo "Processing session: $session_dir"
        
        # Run behavior_pipeline.py for this session
        python3 -u behavior_pipeline.py "$session_dir"
        
        if [ $? -eq 0 ]; then
            echo "Successfully processed session: $session_name"
        else
            echo "Error processing session: $session_name"
        fi
        
        echo "----------------------------------------"
    fi
done

echo "~~~All sessions processed!~~~"