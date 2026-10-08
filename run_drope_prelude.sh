#!/bin/bash
#SBATCH --job-name=drope-prelude-dbg
#SBATCH --account=AIFAC_F02_352
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#
# DroPE recalibration of openeurollm/prelude (Qwen3, ~9B) on a full Booster node
# (4x A100 64GB). ZeRO-3 with the optimizer offloaded to CPU (~110 GB Adam state).
# Defaults to the debug QoS (boost_qos_dbg: max 30 min wall time).
#
# Requires the Qwen3 patch in custom_models/{attention,drope}.py and the model
# downloaded to MODEL_DIR (local folder, so the revision is fixed).
#
# Submit (extra arguments are Hydra overrides passed to launch.sh):
#   sbatch run_drope_prelude.sh max_steps=5 train_batch_size=8 per_device_train_batch_size=1
#
# Full run in the normal queue (command-line flags override the #SBATCH lines):
#   sbatch --qos=normal --time=24:00:00 --job-name=drope-prelude run_drope_prelude.sh
#
# Don't pass dataset_id_or_path / model_name_or_path / model_log_name as overrides:
# they're set below from DATA_DIR / MODEL_DIR / MODEL_LOG_NAME, and Hydra rejects
# a key overridden twice. Change the env vars instead (sbatch --export=ALL,...).
set -euo pipefail

BASE=/leonardo_work/AIFAC_F02_352/anna
DROPE_DIR=${DROPE_DIR:-$BASE/DroPE}
CONDA_ENV=${CONDA_ENV:-drope}
RUN_CFG=${RUN_CFG:-prelude_drope}                         # config name in cfgs/run_cfg/
RUN_CFG=${RUN_CFG%.yaml}
MODEL_DIR=${MODEL_DIR:-$BASE/models/prelude-anneal300b}   # revision anneal300b_iter_0989075
MODEL_LOG_NAME=${MODEL_LOG_NAME:-prelude9b}
DATA_DIR=${DATA_DIR:-$BASE/data/fineweb-edu}
ZERO=${ZERO:-offload_optim}                               # ZeRO-3 + optimizer on CPU
N_GPUS=${SLURM_GPUS_ON_NODE:-4}

# --- Fail fast on missing inputs, before loading anything -------------------
if [ ! -f "$DROPE_DIR/cfgs/run_cfg/$RUN_CFG.yaml" ]; then
    echo "ERROR: config '$RUN_CFG' not found in $DROPE_DIR/cfgs/run_cfg/. Available:" >&2
    ls "$DROPE_DIR/cfgs/run_cfg" | sed 's/\.yaml$//' >&2
    exit 1
fi
if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "ERROR: no config.json in MODEL_DIR=$MODEL_DIR. On the login node run:" >&2
    echo "  hf download openeurollm/prelude --revision anneal300b_iter_0989075 --local-dir $MODEL_DIR" >&2
    exit 1
fi
if ! grep -q "DroPEQwen3ForCausalLM" "$DROPE_DIR/custom_models/drope.py"; then
    echo "ERROR: Qwen3 support missing from custom_models/drope.py (apply the patch)" >&2
    exit 1
fi
N_SHARDS=$(find "$DATA_DIR" -name '*.parquet' 2>/dev/null | wc -l)
if [ "$N_SHARDS" -eq 0 ]; then
    echo "ERROR: no .parquet files under DATA_DIR=$DATA_DIR" >&2
    exit 1
fi
if [ ! -f "$DROPE_DIR/accuracy/accuracy.py" ]; then
    echo "ERROR: evaluate metric missing. On the login node run:" >&2
    echo "  cd $DROPE_DIR && hf download evaluate-metric/accuracy --repo-type space --local-dir accuracy" >&2
    exit 1
fi

# --- Environment ------------------------------------------------------------
set +u                                   # conda/module scripts use unset variables
source "$BASE/miniforge3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# gcc 12 + CUDA 12.6: DeepSpeed JIT-compiles its CPU Adam op for offload_optim
# on the first run (cached afterwards in TORCH_EXTENSIONS_DIR).
module load gcc/12.2.0
module load cuda/12.6
export CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
set -u

export HF_HOME="$BASE/cache/hf"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline WANDB_DIR="$BASE/wandb"
export XDG_CACHE_HOME="$BASE/cache" TRITON_CACHE_DIR="$BASE/cache/triton"
export TORCH_EXTENSIONS_DIR="$BASE/cache/torch_extensions"
export OMP_NUM_THREADS=8
export HYDRA_FULL_ERROR=1

# --- Sanity report ----------------------------------------------------------
echo "Job $SLURM_JOB_ID on $(hostname), $N_GPUS GPUs, config $RUN_CFG, $ZERO"
echo "Model:  $MODEL_DIR ($MODEL_LOG_NAME)"
echo "Data:   $DATA_DIR ($N_SHARDS parquet shards)"
echo "Python: $(command -v python) | gcc: $(gcc -dumpversion) | nvcc: $(command -v nvcc)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch, flash_attn, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, \
'gpus', torch.cuda.device_count(), '| flash_attn', flash_attn.__version__, '| transformers', transformers.__version__)"

# --- Launch -----------------------------------------------------------------
cd "$DROPE_DIR"
./launch.sh "$N_GPUS" "$RUN_CFG" "$ZERO" \
    dataset_id_or_path="$DATA_DIR" \
    model_name_or_path="$MODEL_DIR" \
    model_log_name="$MODEL_LOG_NAME" \
    "$@"