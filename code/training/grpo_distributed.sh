#!/bin/bash
#SBATCH --job-name=grpo_quantum_circuit_gen_multigpu
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --output=../../logs/grpo_%A_%a.out
#SBATCH --error=../../logs/grpo_%A_%a.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=300GB
#SBATCH --gpus=4
#SBATCH --partition=gpu-h200-141g-short
##SBATCH --partition=gpu-debug
##SBATCH --mail-type=BEGIN
##SBATCH --mail-user=linus.jern@aalto.fi

module purge
# module load gcc cuda cmake openmpi
module load scicomp-python-env
module load scicomp-llm-env

source .venv/bin/activate

export WANDB_API_KEY=$(cat .wandb_api_key)

pip install -r requirements.txt

uid="$(date +%Y%m%d_%H%M%S)"

base_model_name="linuzj/quantum-circuit-qubo-3B"
report_to="wandb"

# HYPERPARAMS
epochs=20
block_size=16384
max_prompt_length=4000
temperature=0.95
learning_rate=0.00001

# SAVING
save_strategy='steps'
save_steps=1000

# LOGGING
logging_strategy="steps"
logging_steps=10

# EVAL (options=[no, steps, epoch])
evaluation_strategy="epoch"


# Only do one batch per GPU to reduce memory footprint. Default is 8
per_device_batch_size=1
gradient_accumulation_steps=4

accelerate launch \
    --num_processes=$SLURM_NTASKS_PER_NODE \
    --num_machines=1 \
    --mixed_precision=bf16 \
    --dynamo_backend=no \
    --main_process_port=29501 \
    -- \
    grpo.py \
        --model_name=${base_model_name} \
        --output_dir="data/checkpoints/${uid}" \
        --log_level="info" \
        --max_prompt_length=${max_prompt_length} \
        --temperature=${temperature} \
        --learning_rate=${learning_rate} \
        --block_size=${block_size} \
        --remove_unused_columns=false \
        --logging_strategy=${logging_strategy} \
        --logging_steps=${logging_steps} \
        --evaluation_strategy=${evaluation_strategy} \
        --num_train_epochs=${epochs} \
        --per_device_train_batch_size=${per_device_batch_size} \
        --per_device_eval_batch_size=${per_device_batch_size} \
        --gradient_accumulation_steps=${gradient_accumulation_steps} \
        --bf16=True \
        --report_to=${report_to} \
        --save_strategy=${save_strategy} \
        --save_steps=${save_steps} \
        --save_only_model=True
