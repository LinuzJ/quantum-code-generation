#!/bin/bash
#SBATCH --job-name=instance_eval
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --array=0-579
#SBATCH --output=logs/instance_eval_%A_%a.out
#SBATCH --error=logs/instance_eval_%A_%a.err

set -euo pipefail

module purge
module load scicomp-python-env/2024-01

# Activate local venv if present
if [ -d .venv ]; then
  source .venv/bin/activate
fi

mkdir -p logs out_instance_eval

# You can override these via sbatch --export=ALL,FAKE_BACKEND=FakeSherbrooke,NUM_INSTANCES=580
FAKE_BACKEND=${FAKE_BACKEND:-FakeKyoto}
SIM_MODE=${SIM_MODE:-noisy}
SHOTS=${SHOTS:-2000}
SEED_SIMULATOR=${SEED_SIMULATOR:-0}
NUM_INSTANCES=${NUM_INSTANCES:-580}

IDX=${SLURM_ARRAY_TASK_ID}
if [ "$IDX" -ge "$NUM_INSTANCES" ]; then
  echo "[INFO] IDX=$IDX >= NUM_INSTANCES=$NUM_INSTANCES; exiting."
  exit 0
fi

# Keep things deterministic and avoid oversubscription
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

python3 -u -m src.instance_eval_common \
  --quasar-json generated_circuits/quasar.json \
  --index "$IDX" \
  --fake-backend "$FAKE_BACKEND" \
  --sim-mode "$SIM_MODE" \
  --shots "$SHOTS" \
  --seed-simulator "$SEED_SIMULATOR" \
  --out-dir out_instance_eval
