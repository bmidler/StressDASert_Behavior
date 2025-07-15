#! /bin/bash

# USAGE: sbatch --array=0-N behavior_pipeline.sh <day_directory>
# Where N is the number of sessions minus 1.

#SBATCH -J behavior_pipeline
#SBATCH -p all
#SBATCH -c 1
#SBATCH --mem=32GB
#SBATCH -t 12:00:00
#SBATCH --error=SBATCH_outputs/behavior_pipeline_error_%j_%x_%A_%a_%N_%t.txt
#SBATCH --output=SBATCH_outputs/behavior_pipeline_output_%j_%x_%A_%a_%N_%t.txt

# Check if day directory is provided as argument
if [ $# -eq 0 ]; then
    echo "Usage: sbatch --array=0-N behavior_pipeline.sh <day_directory>"
    echo "Where N is the number of sessions minus 1"
    exit 1
fi

DAY_DIRECTORY=$1

# Validate that the directory exists
if [ ! -d "$DAY_DIRECTORY" ]; then
    echo "Error: Directory $DAY_DIRECTORY does not exist"
    exit 1
fi

# Get all session directories (excluding calibration)
session_dirs=()
for session_dir in "$DAY_DIRECTORY"/*; do
    if [ -d "$session_dir" ]; then
        session_name=$(basename "$session_dir")
        if [ "$session_name" != "calibration" ]; then
            session_dirs+=("$session_dir")
        fi
    fi
done

# Check if SLURM_ARRAY_TASK_ID is set (indicates this is running as part of a job array)
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Error: This script should be run as a SLURM job array"
    echo "Usage: sbatch --array=0-$((${#session_dirs[@]}-1)) behavior_pipeline.sh <day_directory>"
    exit 1
fi

# Get the session directory for this array task
if [ "$SLURM_ARRAY_TASK_ID" -ge "${#session_dirs[@]}" ]; then
    echo "Error: Array task ID $SLURM_ARRAY_TASK_ID is out of range"
    exit 1
fi

SESSION_DIR="${session_dirs[$SLURM_ARRAY_TASK_ID]}"
SESSION_NAME=$(basename "$SESSION_DIR")

module load anacondapy/2023.07-cuda
source activate general
echo "Processing session $((SLURM_ARRAY_TASK_ID + 1)) of ${#session_dirs[@]}: $SESSION_NAME"
echo "Current conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"

# Run behavior_pipeline.py for this session
python3 -u behavior_pipeline.py "$SESSION_DIR"

if [ $? -eq 0 ]; then
    echo "Successfully processed session: $SESSION_NAME"
else
    echo "Error processing session: $SESSION_NAME"
    exit 1
fi

echo "Completed processing session: $SESSION_NAME"