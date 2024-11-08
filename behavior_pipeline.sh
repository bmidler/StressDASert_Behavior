#! /bin/bash

#SBATCH -J behavior_pipeline
#SBATCH -p all

#SBATCH -c 1
#SBATCH --mem=32GB
#SBATCH -t 12:00:00

#SBATCH --error=SBATCH_outputs/behavior_pipeline_error_%j_%x_%A_%a_%N_%t.txt
#SBATCH --output=SBATCH_outputs/behavior_pipeline_output_%j_%x_%A_%a_%N_%t.txt

module load anacondapy/2023.07-cuda
source activate general
echo "Starting behavior pipeline."
echo "Current conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"

# Run behavior_pipeline.py.
python3 -u behavior_pipeline.py

echo "~~~All done!~~~"