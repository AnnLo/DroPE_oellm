#!/bin/bash
# Submit prelude ablation arms (default: B = DroPE, C = Soft DroPE; both keep Qwen3's QK norm), all on
# the long-document mix, each as a chain of N_JOBS resumable 24 h jobs on NODES nodes.
#
#   ./submit_ablations.sh [NODES=8] [N_JOBS=3]
#
# NODES must divide 512 / (2 per GPU x 4 GPUs) = 64:  1, 2, 4, 8, 16, 32 or 64.
# Uses run_drope_prelude_multinode.sh (fixed RUN_ID output dir -> resume, hpZ by default).
set -euo pipefail
NODES=${1:-8}
N_JOBS=${2:-3}
DATA_DIR=${DATA_DIR:-/leonardo_scratch/fast/AIFAC_F02_352/anna/data/longctx-mix-longdoc}
cd "$(dirname "$0")"
(( 64 % NODES == 0 )) || { echo "NODES=$NODES does not divide 64" >&2; exit 1; }
[ -n "$(ls "$DATA_DIR"/train-*.parquet 2>/dev/null)" ] || { echo "no shards in $DATA_DIR" >&2; exit 1; }

ARMS=${ARMS:-"B_drope:prelude_drope_longdoc C_soft:prelude_soft_drope_longdoc"}
for pair in $ARMS; do
    RUN_ID="abl_${pair%%:*}"; CFG=${pair#*:}
    [ -f "cfgs/run_cfg/$CFG.yaml" ] || { echo "missing cfgs/run_cfg/$CFG.yaml" >&2; exit 1; }
    prev=""
    for i in $(seq 1 "$N_JOBS"); do
        dep=${prev:+--dependency=afterany:$prev}
        prev=$(sbatch --parsable $dep --nodes="$NODES" --job-name="$RUN_ID" \
               --export=ALL,RUN_ID="$RUN_ID",RUN_CFG="$CFG",DATA_DIR="$DATA_DIR" \
               run_drope_prelude_multinode.sh)
        echo "$RUN_ID ($CFG)  job $i/$N_JOBS: $prev"
    done
done