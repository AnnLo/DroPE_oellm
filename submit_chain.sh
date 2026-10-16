#!/bin/bash
# Submit N chained 24 h jobs; each starts after the previous one ends (any reason)
# and resumes from the newest checkpoint. Finished-early jobs exit immediately.
#   ./submit_chain.sh 5                       # default run
#   RUN_ID=10b_run2 ./submit_chain.sh 5       # extra env vars are passed through
set -euo pipefail
N=${1:-5}
SCRIPT=${SCRIPT:-run_drope_prelude_multinode.sh}
EXPORTS="ALL${RUN_ID:+,RUN_ID=$RUN_ID}${RUN_CFG:+,RUN_CFG=$RUN_CFG}${DATA_DIR:+,DATA_DIR=$DATA_DIR}"
jid=$(sbatch --parsable --export="$EXPORTS" "$SCRIPT")
echo "job 1: $jid"
for i in $(seq 2 "$N"); do
  jid=$(sbatch --parsable --export="$EXPORTS" --dependency=afterany:"$jid" "$SCRIPT")
  echo "job $i: $jid"
done
echo "Cancel the rest of the chain with: scancel <jobid> (pending jobs listed by squeue -u \$USER)"