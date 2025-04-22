#!/bin/bash

#SBATCH --job-name=eval_suite
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  
#SBATCH --cpus-per-task=16   # Adjust as needed
#SBATCH --gres=gpu:a40:6     # Adjust if you need more or different GPUs
#SBATCH --qos=short
#SBATCH --partition=nlprx-lab
#SBATCH --output=eval-logs/%x-%j.out

set -x -e

echo "START TIME: $(date)"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"

# Activate Conda environment
source /srv/nlprx-lab/share6/gramesh31/miniconda3/etc/profile.d/conda.sh
# source /nethome/gramesh31/.bashrc  # Ensure conda is initialized properly
echo $HF_HOME
conda activate s1p

# Set logging path
LOG_PATH="eval-logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.log"

# Define combinations of parameters
MAX_DEPTHS=(4 8 12)
BEAM_WIDTHS=(2 4)
IDEAS=(3)
DATASETS=("aime")

# Loop through combinations and execute the command
for MAX_DEPTH in "${MAX_DEPTHS[@]}"; do
    for BEAM_WIDTH in "${BEAM_WIDTHS[@]}"; do
        for IDEA in "${IDEAS[@]}"; do
            for DATASET in "${DATASETS[@]}"; do
                CMD="\
python eval_suite.py \
     --max_depth=${MAX_DEPTH} \
     --beam_width=${BEAM_WIDTH} \
     --ideas=${IDEA} \
     --dataset=${DATASET}
"

                echo "Running command: $CMD"
                srun --kill-on-bad-exit=1 bash -c "$CMD" 2>&1 | tee -a $LOG_PATH
            done
        done
    done
done

echo "END TIME: $(date)"