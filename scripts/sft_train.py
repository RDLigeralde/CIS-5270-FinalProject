import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    STUDENT_MODEL,
    SFT_TRAIN_FILE, SFT_VAL_FILE,
    SFT_EPOCHS, BATCH_SIZE, LR_MULTIPLIER,
    DATA_DIR, WANDB_PROJECT,
)
from scripts.utils import get_openai_client, upload_file, wait_for_job, log_job_result, WandbLogger

MODEL_ID_PATH = f"{DATA_DIR}/sft_model_id.txt"


def run_sft(wait: bool = True, no_wandb: bool = False, project: str = WANDB_PROJECT,
            experiment: str | None = None) -> str:
    client = get_openai_client()

    wb = WandbLogger(
        project=project,
        experiment=experiment or "sft",
        config={
            "job_type": "sft",
            "base_model": STUDENT_MODEL,
            "n_epochs": SFT_EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr_multiplier": LR_MULTIPLIER,
        },
        disabled=no_wandb,
    )

    print("=== SFT Warm-start ===")
    train_id = upload_file(client, SFT_TRAIN_FILE)
    val_id = upload_file(client, SFT_VAL_FILE)

    print(f"\nCreating supervised fine-tuning job for {STUDENT_MODEL}...")
    job = client.fine_tuning.jobs.create(
        model=STUDENT_MODEL,
        training_file=train_id,
        validation_file=val_id,
        method={
            "type": "supervised",
            "supervised": {
                "hyperparameters": {
                    "n_epochs": SFT_EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "learning_rate_multiplier": LR_MULTIPLIER,
                }
            },
        },
        extra_body={"trainingType": "GlobalStandard"},
        suffix="sft-warmstart",
    )
    print(f"Job created: {job.id}  status={job.status}")
    wb.log({"job/id": job.id})

    if not wait:
        print(f"Run with --wait to poll until completion, or check status manually.")
        wb.finish()
        return job.id

    completed = wait_for_job(client, job.id, wandb_logger=wb)
    if completed.status != "succeeded":
        wb.finish()
        raise RuntimeError(f"SFT job {job.id} ended with status: {completed.status}")

    model_id = completed.fine_tuned_model
    print(f"\nSFT complete. Fine-tuned model: {model_id}")

    log_job_result(client, completed, wb)
    wb.finish()

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(MODEL_ID_PATH, "w") as f:
        f.write(model_id)
    print(f"Model ID saved -> {MODEL_ID_PATH}")

    return model_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", action="store_true", default=True,
                        help="Poll until the job completes (default: True)")
    parser.add_argument("--no-wait", dest="wait", action="store_false")
    parser.add_argument("--no-wandb", action="store_true", default=False,
                        help="Disable Weights & Biases logging")
    parser.add_argument("--project", default=WANDB_PROJECT, help="W&B project name")
    parser.add_argument("--experiment", default=None, help="W&B run name")
    args = parser.parse_args()
    run_sft(wait=args.wait, no_wandb=args.no_wandb,
            project=args.project, experiment=args.experiment)
