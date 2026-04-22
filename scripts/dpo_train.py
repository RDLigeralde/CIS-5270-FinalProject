import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    STUDENT_MODEL,
    DPO_TRAIN_FILE, DPO_VAL_FILE,
    DPO_EPOCHS, BATCH_SIZE, LR_MULTIPLIER,
    DATA_DIR, WANDB_PROJECT,
)
from scripts.utils import get_openai_client, upload_file, wait_for_job, log_job_result, WandbLogger

SFT_MODEL_ID_PATH = f"{DATA_DIR}/sft_model_id.txt"
DPO_MODEL_ID_PATH = f"{DATA_DIR}/dpo_model_id.txt"


def _load_model_id(path: str, fallback: str) -> str:
    if os.path.exists(path):
        with open(path) as f:
            mid = f.read().strip()
        if mid:
            return mid
    print(f"Warning: {path} not found, using fallback base model: {fallback}")
    return fallback


def run_dpo(base_model: str | None = None, wait: bool = True, no_wandb: bool = False,
            project: str = WANDB_PROJECT, experiment: str | None = None) -> str:
    client = get_openai_client()

    if base_model is None:
        base_model = _load_model_id(SFT_MODEL_ID_PATH, STUDENT_MODEL)

    wb = WandbLogger(
        project=project,
        experiment=experiment or "dpo",
        config={
            "job_type": "dpo",
            "base_model": base_model,
            "n_epochs": DPO_EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr_multiplier": LR_MULTIPLIER,
        },
        disabled=no_wandb,
    )
    wb.define_dpo_charts()

    print("=== DPO Training ===")
    print(f"Base model: {base_model}")

    train_id = upload_file(client, DPO_TRAIN_FILE)
    val_id = upload_file(client, DPO_VAL_FILE)

    print(f"\nCreating DPO fine-tuning job...")
    job = client.fine_tuning.jobs.create(
        model=base_model,
        training_file=train_id,
        validation_file=val_id,
        method={
            "type": "dpo",
            "dpo": {
                "hyperparameters": {
                    "n_epochs": DPO_EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "learning_rate_multiplier": LR_MULTIPLIER,
                }
            },
        },
        extra_body={"trainingType": "GlobalStandard"},
        suffix="dpo-style-correctness",
    )
    print(f"Job created: {job.id}  status={job.status}")
    wb.log({"job/id": job.id})

    if not wait:
        wb.finish()
        return job.id

    completed = wait_for_job(client, job.id, wandb_logger=wb)
    if completed.status != "succeeded":
        wb.finish()
        raise RuntimeError(f"DPO job {job.id} ended with status: {completed.status}")

    model_id = completed.fine_tuned_model
    print(f"\nDPO complete. Fine-tuned model: {model_id}")

    log_job_result(client, completed, wb)
    wb.finish()

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(DPO_MODEL_ID_PATH, "w") as f:
        f.write(model_id)
    print(f"Model ID saved -> {DPO_MODEL_ID_PATH}")

    return model_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=None,
                        help="Base model ID (default: reads sft_model_id.txt)")
    parser.add_argument("--wait", action="store_true", default=True)
    parser.add_argument("--no-wait", dest="wait", action="store_false")
    parser.add_argument("--no-wandb", action="store_true", default=False,
                        help="Disable Weights & Biases logging")
    parser.add_argument("--project", default=WANDB_PROJECT, help="W&B project name")
    parser.add_argument("--experiment", default=None, help="W&B run name")
    args = parser.parse_args()
    run_dpo(base_model=args.base_model, wait=args.wait, no_wandb=args.no_wandb,
            project=args.project, experiment=args.experiment)
