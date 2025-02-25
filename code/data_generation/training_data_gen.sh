#!/bin/bash
#SBATCH --job-name=trainingdata_gen_batch
#SBATCH --time=02:00:00
#SBATCH --output=../../logs/hypermaxcut_%A_%a.out
#SBATCH --error=../../logs/hypermaxcut_%A_%a.err
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=50GB
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=linus.jern@aalto.fi

module purge
module load gcc cuda cmake openmpi
module load scicomp-python-env/2024-01
module load scicomp-llm-env

source .venv/bin/activate
pip install -r requirements.txt
pip install custatevec_cu12
pip install pennylane-lightning-gpu

PROBLEMS=("hypermaxcut" "community_detection" "graph_coloring" "connected_components")
ANSATZ_OPTIONS=(1 5 12 13)

# Get counts
NUM_ANSATZ=${#ANSATZ_OPTIONS[@]}   # 4
NUM_PROBLEMS=${#PROBLEMS[@]}       # 4

PROBLEM_INDEX=$(( SLURM_ARRAY_TASK_ID / NUM_ANSATZ ))
ANSATZ_INDEX=$(( SLURM_ARRAY_TASK_ID % NUM_ANSATZ ))

SELECTED_PROBLEM=${PROBLEMS[$PROBLEM_INDEX]}
SELECTED_ANSATZ=${ANSATZ_OPTIONS[$ANSATZ_INDEX]}

echo "Running ${SELECTED_PROBLEM} with ansatz option ${SELECTED_ANSATZ}..."

layers=1
output_dir="out/"

python3 -m src.main \
    --problem ${SELECTED_PROBLEM} \
    --layers ${layers} \
    --ansatz_template ${SELECTED_ANSATZ} \
    --output_path="${output_dir}"