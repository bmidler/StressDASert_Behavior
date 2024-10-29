#! /bin/bash

#SBATCH -J behavior_pipeline
#SBATCH -p all

#SBATCH -c 1
#SBATCH --mem=32GB
#SBATCH -t 02:00:00

#SBATCH --error=SBATCH_outputs/behavior_pipeline_error.txt
#SBATCH --output=SBATCH_outputs/behavior_pipeline_output.txt

module load anacondapy/2023.07-cuda
source activate sleap
echo "Starting behavior pipeline."
echo "Current conda environment: $(conda info --envs | grep '*' | awk '{print $1}')"

# Run behavior_pipeline.py.
python3 -u behavior_pipeline.py

echo "~~~All done!~~~"