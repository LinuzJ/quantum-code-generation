#!/bin/bash
#SBATCH --job-name=sft_quantum_circuit_gen_multigpu
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --mem=400GB
#SBATCH --cpus-per-task=32
#SBATCH --gpus=8
#SBATCH --partition=gpu-h200-141g-ellis


module purge
module load scicomp-python-env/2024-01
module load scicomp-llm-env

source .venv/bin/activate

export WANDB_API_KEY=$(cat .wandb_api_key)

pip install -r requirements.txt


uid="$(date +%Y%m%d_%H%M%S)"

gpus=8
nodes=1

base_model_name="Qwen/Qwen3-8B"
output_dir_name="cong/sft_quantum_circuit_gen_8B_${uid}"

epochs=15
block_size=16384
save_strategy='steps'
save_steps=24000


# Only do one batch per GPU to reduce memory footprint. Default is 8
per_device_batch_size=1
gradient_accumulation_steps=1

accelerate launch \
        --use_fsdp \
        --config_file "fsdp_config.yaml" \
        --mixed_precision "bf16" \
        --num_processes ${gpus} \
        --num_machines ${nodes} \
        -- \
        sft.py \
                --model_name=${base_model_name} \
                --output_dir=${output_dir_name} \
                --log_level="info" \
                --block_size=${block_size} \
                --num_train_epochs=${epochs} \
                --per_device_train_batch_size=${per_device_batch_size} \
                --per_device_eval_batch_size=${per_device_batch_size} \
                --gradient_accumulation_steps=${gradient_accumulation_steps} \
                --fsdp="full_shard auto_wrap" \
                --fsdp_config="fsdp_config.json" \
                --bf16=True \
                --save_strategy=${save_strategy} \
                --save_steps=${save_steps} \
                --push_to_hub=True \
                --hub_strategy=all_checkpoints
