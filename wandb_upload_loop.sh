#!/bin/bash
#SBATCH --job-name=wandb-upload
#SBATCH --account=AIFAC_F02_352
#SBATCH --partition=lrd_all_serial
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/leonardo_work/AIFAC_F02_352/anna/DroPE/wandb-upload-%j.out
# Uploads the training curves from the .out logs to W&B every 5 min; resubmits itself
# before the 4 h limit. Stop with: scancel -n wandb-upload
cd /leonardo_work/AIFAC_F02_352/anna/DroPE
source /leonardo_work/AIFAC_F02_352/anna/env.sh && conda activate drope
end=$(( $(date +%s) + 3*3600 + 45*60 ))
while [ "$(date +%s)" -lt "$end" ]; do
  python upload_log_to_wandb.py --logs 'abl_B_drope-*.out' --id abl_B_drope_log --name B_drope
  python upload_log_to_wandb.py --logs 'abl_C_soft-*.out'  --id abl_C_soft_log  --name C_soft_drope --alpha_end_step 2500
  python upload_log_to_wandb.py --logs 'abl_D_short-*.out' --id abl_D_short_log --name D_soft_output_short --alpha_end_step 250
  date; sleep 300
done
sbatch /leonardo_work/AIFAC_F02_352/anna/DroPE/wandb_upload_loop.sh
