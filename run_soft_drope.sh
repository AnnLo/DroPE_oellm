#!/bin/bash
#SBATCH --job-name=soft-drope-qknorm-smollm360m-dbg
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
# Soft DroPE (gradual RoPE removal) on SmolLM-360M, one Booster node (4x A100 64GB).
# Defaults to the debug QoS (30 min). Extra arguments are Hydra overrides.
#
#   sbatch run_soft_drope.sh max_steps=20                                   # angle, debug
#   sbatch run_soft_drope.sh max_steps=20 model_custom_config.soft_mode=output
#   sbatch --export=ALL,RUN_CFG=smollm_360m_drope_qk_norm run_soft_drope.sh max_steps=20   # abrupt baseline
#
# Full run in the normal queue (command-line flags override the #SBATCH lines):
#   sbatch --qos=normal --time=24:00:00 --job-name=soft-drope-smollm360m run_soft_drope.sh max_steps=...
#
# Schedule overrides:  trainer.callbacks.0.end_frac=0.3  trainer.callbacks.0.shape=cosine
# Don't pass dataset_id_or_path=...: it's set from DATA_DIR (Hydra rejects duplicate keys).
set -euo pipefail

BASE=/leonardo_work/AIFAC_F02_352/anna
DROPE_DIR=${DROPE_DIR:-$BASE/DroPE_soft}
CONDA_ENV=${CONDA_ENV:-drope}
RUN_CFG=${RUN_CFG:-smollm_360m_soft_drope_qk_norm}; RUN_CFG=${RUN_CFG%.yaml}
DATA_DIR=${DATA_DIR:-$BASE/data/fineweb-edu}
ZERO=${ZERO:-zero1}                       # 360M fits easily without sharding
N_GPUS=${SLURM_GPUS_ON_NODE:-4}

# --- Fail fast ---------------------------------------------------------------
[ -f "$DROPE_DIR/cfgs/run_cfg/$RUN_CFG.yaml" ] || { echo "ERROR: no config $RUN_CFG in $DROPE_DIR/cfgs/run_cfg" >&2; exit 1; }
[ -f "$DROPE_DIR/custom_models/soft_drope.py" ] || { echo "ERROR: custom_models/soft_drope.py missing" >&2; exit 1; }
[ -f "$DROPE_DIR/accuracy/accuracy.py" ] || { echo "ERROR: accuracy metric missing (ln -s ../DroPE/accuracy accuracy)" >&2; exit 1; }
N_SHARDS=$(find "$DATA_DIR" -name '*.parquet' 2>/dev/null | wc -l)
[ "$N_SHARDS" -gt 0 ] || { echo "ERROR: no .parquet files under $DATA_DIR" >&2; exit 1; }

# --- Environment ---------------------------------------------------------------
set +u
source "$BASE/miniforge3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
module load gcc/12.2.0 cuda/12.6
export CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
set -u

export HF_HOME="$BASE/cache/hf"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline WANDB_DIR="$BASE/wandb"
export XDG_CACHE_HOME="$BASE/cache" TRITON_CACHE_DIR="$BASE/cache/triton"
export TORCH_EXTENSIONS_DIR="$BASE/cache/torch_extensions"
export OMP_NUM_THREADS=8 HYDRA_FULL_ERROR=1

echo "Job $SLURM_JOB_ID on $(hostname), $N_GPUS GPUs, config $RUN_CFG, $ZERO"
echo "Code: $DROPE_DIR ($(git -C "$DROPE_DIR" rev-parse --abbrev-ref HEAD) @ $(git -C "$DROPE_DIR" rev-parse --short HEAD))"
echo "Data: $DATA_DIR ($N_SHARDS parquet shards)"

# --- Launch --------------------------------------------------------------------
cd "$DROPE_DIR"
./launch.sh "$N_GPUS" "$RUN_CFG" "$ZERO" dataset_id_or_path="$DATA_DIR" "$@"