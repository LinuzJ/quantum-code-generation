#!/bin/bash
#SBATCH --job-name=speed_bench
#SBATCH --time=04:20:00
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --output=logs/speed_bench_%A.out
#SBATCH --error=logs/speed_bench_%A.err

set -euo pipefail

module purge
module load scicomp-python-env/2024-01

if [ -d .venv ]; then
  source .venv/bin/activate
fi

mkdir -p logs

# --------- Config (override via sbatch --export=ALL,VAR=...) ---------
OUT_DIR=${OUT_DIR:-out_speed}
QUASAR_JSON=${QUASAR_JSON:-generated_circuits/quasar.json}

# Subset selection (default: small enough for ~10-minute quota)
K=${K:-3}
SEED=${SEED:-0}
INDICES_FILE=${INDICES_FILE:-}

# Local simulation settings
FAKE_BACKEND=${FAKE_BACKEND:-FakeTorino}
SHOTS=${SHOTS:-100}
SEED_SIMULATOR=${SEED_SIMULATOR:-0}
LOCAL_NO_SIM=${LOCAL_NO_SIM:-0}

# Runtime settings (defaults compare FakeTorino vs ibm_torino)
RUNTIME_CHANNEL=${RUNTIME_CHANNEL:-ibm_cloud}
RUNTIME_BACKEND=${RUNTIME_BACKEND:-ibm_torino}
RUNTIME_INSTANCE=${RUNTIME_INSTANCE:-}

# If set to 1, only do local timings (no IBM Runtime submission)
SKIP_RUNTIME=${SKIP_RUNTIME:-0}

# If set to 1, compile against runtime backend but do not submit (physical-duration only)
RUNTIME_NO_SUBMIT=${RUNTIME_NO_SUBMIT:-0}

mkdir -p "$OUT_DIR"

if [ -z "$INDICES_FILE" ]; then
  INDICES_FILE="$OUT_DIR/indices_k${K}_seed${SEED}.txt"
  python3 -u -m src.make_instance_subset \
    --k "$K" \
    --seed "$SEED" \
    --out "$INDICES_FILE"
fi

EXTRA_RUNTIME_ARGS=()
if [ -n "$RUNTIME_INSTANCE" ]; then
  EXTRA_RUNTIME_ARGS+=(--runtime-instance "$RUNTIME_INSTANCE")
fi

EXTRA_SKIP_ARGS=()
if [ "$SKIP_RUNTIME" = "1" ]; then
  EXTRA_SKIP_ARGS+=(--skip-runtime)
fi

EXTRA_NO_SUBMIT_ARGS=()
if [ "$RUNTIME_NO_SUBMIT" = "1" ]; then
  EXTRA_NO_SUBMIT_ARGS+=(--runtime-no-submit)
fi

EXTRA_LOCAL_NO_SIM_ARGS=()
if [ "$LOCAL_NO_SIM" = "1" ]; then
  EXTRA_LOCAL_NO_SIM_ARGS+=(--local-no-sim)
fi

python3 -u -m src.benchmark_execution_speed \
  --quasar-json "$QUASAR_JSON" \
  --indices-file "$INDICES_FILE" \
  --fake-backend "$FAKE_BACKEND" \
  --shots "$SHOTS" \
  --seed-simulator "$SEED_SIMULATOR" \
  --runtime-channel "$RUNTIME_CHANNEL" \
  --runtime-backend "$RUNTIME_BACKEND" \
  "${EXTRA_RUNTIME_ARGS[@]}" \
  "${EXTRA_SKIP_ARGS[@]}" \
  "${EXTRA_NO_SUBMIT_ARGS[@]}" \
  "${EXTRA_LOCAL_NO_SIM_ARGS[@]}" \
  --out-dir "$OUT_DIR"
