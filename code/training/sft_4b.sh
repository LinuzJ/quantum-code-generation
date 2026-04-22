#!/bin/bash
#SBATCH --job-name=sft_qcg_multigpu
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --mem=400GB
#SBATCH --cpus-per-task=32
#SBATCH --gpus=8
#SBATCH --partition=gpu-h200-141g-ellis

set -euo pipefail

module purge
module load scicomp-python-env/2024-01
module load scicomp-llm-env
source .venv/bin/activate

# ---- Auth / caches ----
export BASE_DIR="$(pwd)"
export WANDB_API_KEY="$(cat .wandb_api_key)"
export HF_TOKEN="${HF_TOKEN:-$(cat .hf_token 2>/dev/null || true)}"
export HF_HOME="${BASE_DIR}/hf"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1


uid="$(date +%Y%m%d_%H%M%S)"
gpus=8
nodes=1

base_model_name="Qwen/Qwen3-4B-Instruct-2507"

# Absolute, shared output path visible to all ranks
output_dir_abs="${BASE_DIR}/experiments/sft_quantum_circuit_gen_4B_${uid}"

# Fixed HF repo name (no timestamps)
hub_model_id="Benyucong/sft_quantum_circuit_gen_4B"

epochs=15
block_size=16384
save_strategy='steps'
save_steps=24000
save_total_limit=1  # optional: keep last 1 on hub/local

per_device_batch_size=1
gradient_accumulation_steps=1

hub_strategy=end

accelerate launch \
  --use_fsdp \
  --config_file fsdp_config.yaml \
  --num_processes ${gpus} \
  --num_machines ${nodes} \
  --mixed_precision bf16 \
  -- \
  sft_4b.py \
    --model_name=${base_model_name} \
    --output_dir=${output_dir_abs} \
    --hub_model_id=${hub_model_id} \
    --push_to_hub=False \
    --hub_strategy=${hub_strategy} \
    --save_total_limit=${save_total_limit} \
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
    --save_steps=${save_steps}
