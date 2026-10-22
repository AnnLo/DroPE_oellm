#!/bin/bash
#SBATCH --job-name=drope-prelude-abl
#SBATCH --account=AIFAC_F02_352
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#
# Multi-node DroPE recalibration of prelude (Qwen3, ~9B): 4 nodes x 4 A100-64GB = 16 GPUs,
# DeepSpeed ZeRO-3 without offload. Resumable: every job continues from the newest
# checkpoint in a FIXED output dir (RUN_ID), so jobs can be chained (submit_chain.sh).
# If training already finished (final model saved), the job exits immediately.
#
# Submit one job:   sbatch run_drope_prelude_multinode.sh
# Submit a chain:   ./submit_chain.sh 5            (5 x 24h jobs, each waits for the previous)
# Quick 2-step multi-node smoke test (debug queue, 30 min):
#   sbatch --qos=boost_qos_dbg --time=00:30:00 --export=ALL,RUN_ID=smoketest \
#          run_drope_prelude_multinode.sh max_steps=2
#
# Env vars (sbatch --export=ALL,VAR=...):
#   RUN_CFG   (prelude_drope)  run config in cfgs/run_cfg/
#   RUN_ID    (10b_run1)       fixed name of the output dir -> enables resume; change for a new run
#   DATA_DIR  (scratch longctx-mix)
#   RESULTS   (scratch results)
#   SAVE_STEPS (100)           ~1.7 h between checkpoints at 4 nodes; at most this much is lost on timeout
#   ACCEL_CFG  (accelerate_configs/deepspeed_zero3.yaml)  e.g. accelerate_configs/deepspeed_zero3_hpz.yaml
set -eo pipefail

BASE=/leonardo_work/AIFAC_F02_352/anna
SCR=/leonardo_scratch/fast/AIFAC_F02_352/anna
DROPE_DIR=${DROPE_DIR:-$BASE/DroPE}
CONDA_ENV=${CONDA_ENV:-drope}
RUN_CFG=${RUN_CFG:-prelude_drope}; RUN_CFG=${RUN_CFG%.yaml}
RUN_ID=${RUN_ID:-10b_run1}
DATA_DIR=${DATA_DIR:-$SCR/data/longctx-mix}
RESULTS=${RESULTS:-$SCR/results}
SAVE_STEPS=${SAVE_STEPS:-100}
ACCEL_CFG=${ACCEL_CFG:-accelerate_configs/deepspeed_zero3_hpz.yaml}   # ZeRO++ hpZ (~30% faster multi-node)
GPUS_PER_NODE=4
NNODES=$SLURM_JOB_NUM_NODES
NPROC=$((NNODES * GPUS_PER_NODE))

# --- Fail fast ---------------------------------------------------------------
[ -f "$DROPE_DIR/cfgs/run_cfg/$RUN_CFG.yaml" ] || { echo "ERROR: no config $RUN_CFG" >&2; exit 1; }
N_SHARDS=$(find "$DATA_DIR" -maxdepth 1 -name 'train-*.parquet' | wc -l)
[ "$N_SHARDS" -gt 0 ] || { echo "ERROR: no shards in $DATA_DIR" >&2; exit 1; }
[ -f "$DROPE_DIR/accuracy/accuracy.py" ] || { echo "ERROR: accuracy metric missing" >&2; exit 1; }
grep -q DroPEQwen3ForCausalLM "$DROPE_DIR/custom_models/drope.py" || { echo "ERROR: Qwen3 patch missing" >&2; exit 1; }

# --- Environment (inherited by srun tasks on all nodes) ----------------------
set +u
source "$BASE/miniforge3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
module load gcc/12.2.0 cuda/12.6
export CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
export HF_HOME="$BASE/cache/hf" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline WANDB_DIR="$SCR/wandb" WANDB_RUN_ID="drope-$RUN_ID" WANDB_RESUME=allow
export XDG_CACHE_HOME="$BASE/cache" TRITON_CACHE_DIR="$SCR/cache/triton"
export TORCH_EXTENSIONS_DIR="$BASE/cache/torch_extensions"
export OMP_NUM_THREADS=8 HYDRA_FULL_ERROR=1 NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # less fragmentation with large logits
mkdir -p "$WANDB_DIR" "$TRITON_CACHE_DIR" "$RESULTS"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 1000))
export MASTER_ADDR MASTER_PORT

# Fixed output dir => the trainer finds and resumes from the newest checkpoint-*.
OUT_DIR="$RESULTS/$RUN_ID"
if [ -f "$OUT_DIR/config.json" ] && ls "$OUT_DIR"/*.safetensors >/dev/null 2>&1; then
    echo "Final model already in $OUT_DIR - training finished, nothing to do."
    exit 0
fi
LAST_CKPT=$(ls -d "$OUT_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)

echo "Job $SLURM_JOB_ID: $NNODES nodes x $GPUS_PER_NODE GPUs = $NPROC procs, master $MASTER_ADDR:$MASTER_PORT"
echo "Accelerate/DeepSpeed config: $ACCEL_CFG"
echo "Config $RUN_CFG | data $DATA_DIR ($N_SHARDS shards) | out $OUT_DIR"
echo "Resuming from: ${LAST_CKPT:-<none, fresh start>}"
python -c "import torch, flash_attn, transformers, deepspeed; print('torch', torch.__version__, '| flash_attn', flash_attn.__version__, '| transformers', transformers.__version__, '| deepspeed', deepspeed.__version__)"

# Hydra overrides (quoted safely for the remote bash -c)
OVERRIDES=(
  "run_cfg@_global_=$RUN_CFG"
  "dataset_id_or_path=$DATA_DIR"
  "output_dir=$OUT_DIR"
  "save_steps=$SAVE_STEPS"
  "$@"
)
OVR=$(printf '%q ' "${OVERRIDES[@]}")

cd "$DROPE_DIR"
srun --overlap --nodes="$NNODES" --ntasks-per-node=1 --kill-on-bad-exit=1 bash -c "
  accelerate launch \
    --config_file $ACCEL_CFG \
    --num_machines $NNODES --num_processes $NPROC \
    --machine_rank \$SLURM_NODEID \
    --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
    --rdzv_backend static --same_network \
    train.py $OVR
"