"""Incrementally upload a (chained) run's training curve from its .out logs to W&B.
Only steps not yet in the W&B run are uploaded, so it is safe to run repeatedly.
  python upload_log_to_wandb.py --logs 'abl_C_soft-*.out' --id abl_C_soft_log --name C_soft_drope --alpha_end_step 2500
"""
import argparse, ast, glob, re
import wandb

p = argparse.ArgumentParser()
p.add_argument("--logs", required=True)
p.add_argument("--id", required=True)
p.add_argument("--name", required=True)
p.add_argument("--project", default="drope-prelude-ablations")
p.add_argument("--entity", default=None)
p.add_argument("--logging_steps", type=int, default=5)
p.add_argument("--alpha_end_step", type=float, default=None)
a = p.parse_args()

def job_id(path):
    m = re.search(r"-(\d+)\.out$", path)
    return int(m.group(1)) if m else 0

points = {}
for f in sorted(glob.glob(a.logs), key=job_id):
    text = open(f, errors="replace").read()
    m = re.search(r"Resuming from: .*checkpoint-(\d+)", text)
    start = int(m.group(1)) if m else 0
    for i, s in enumerate(re.findall(r"\{'loss': [^}]*\}", text)):
        try:
            points[start + (i + 1) * a.logging_steps] = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            pass
if not points:
    raise SystemExit(f"no loss entries in {a.logs}")

run = wandb.init(project=a.project, entity=a.entity, id=a.id, name=a.name, resume="allow")
wandb.define_metric("train/global_step")
wandb.define_metric("train/*", step_metric="train/global_step")
done = int(run.summary.get("train/global_step", 0) or 0)
new = [s for s in sorted(points) if s > done]
for step in new:
    out = {f"train/{k}": v for k, v in points[step].items()}
    out["train/global_step"] = step
    if a.alpha_end_step:
        out["train/rope_alpha"] = max(0.0, 1.0 - step / a.alpha_end_step)
    wandb.log(out)
run.finish()
print(f"{a.id}: {len(new)} new points (up to step {max(points)}), {run.url}")
