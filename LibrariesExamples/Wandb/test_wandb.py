"""Smoke test for Weights & Biases (CoreWeave Forge) logging.

Set credentials via environment (do not hardcode API keys):

  setx WANDB_API_KEY "<your-forge-api-key>"
  setx WANDB_ENTITY "<your-entity>"
  setx WANDB_PROJECT "PhoDibaBatch"

  export WANDB_API_KEY="<your-forge-api-key>"
  export WANDB_ENTITY="<your-entity>"
  export WANDB_PROJECT="PhoDibaBatch"

Optional: `wandb login --cloud` for Forge multi-tenant cloud.
See https://docs.coreweave.com/products/wandb/quickstart
"""
import os
from pathlib import Path

import wandb

# Use disabled mode when no API key is present so import/CI smoke tests stay offline-safe.
mode = "online" if os.environ.get("WANDB_API_KEY") else "disabled"
project = os.environ.get("WANDB_PROJECT", "PhoDibaBatch")
entity = os.environ.get("WANDB_ENTITY")

init_kwargs = dict(project=project, job_type="smoke", mode=mode, reinit=True)
if entity:
    init_kwargs["entity"] = entity

with wandb.init(**init_kwargs) as run:
    run.config.update({"computer": "Apogee", "smoke_test": True}, allow_val_change=True)
    run.summary["status"] = "ok"
    test_file_path = Path("README.md")
    if test_file_path.exists():
        artifact = wandb.Artifact(name="smoke-readme", type="output")
        artifact.add_file(str(test_file_path))
        run.log_artifact(artifact)

print(f"wandb smoke test finished (mode={mode})")
