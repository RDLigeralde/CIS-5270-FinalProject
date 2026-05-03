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
            project: str = WANDB_PROJECT, experiment: str | None = None,
            n_epochs: int | None = None,
            batch_size: int | None = None,
            lr_multiplier: float | None = None) -> str:
    client = get_openai_client()

    if base_model is None:
        base_model = _load_model_id(SFT_MODEL_ID_PATH, STUDENT_MODEL)

    n_epochs = DPO_EPOCHS if n_epochs is None else n_epochs
    batch_size = BATCH_SIZE if batch_size is None else batch_size
    lr_multiplier = LR_MULTIPLIER if lr_multiplier is None else lr_multiplier

    wb = WandbLogger(
        project=project,
        experiment=experiment or "dpo",
        config={
            "job_type": "dpo",
            "base_model": base_model,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "lr_multiplier": lr_multiplier,
        },
        disabled=no_wandb,
    )
    wb.define_dpo_charts()

    print("=== DPO Training ===")
    print(f"Base model: {base_model}")
    print(f"Hyperparameters: epochs={n_epochs}, batch_size={batch_size}, lr_multiplier={lr_multiplier}")

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
                    "n_epochs": n_epochs,
                    "batch_size": batch_size,
                    "learning_rate_multiplier": lr_multiplier,
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
        print(f"\nJob failed. Fetching events for {job.id}...")
        for ev in client.fine_tuning.jobs.list_events(job.id, limit=20):
            print(f"  [{getattr(ev, 'level', '?')}] {getattr(ev, 'message', ev)}")
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
    parser.add_argument("--epochs", type=int, default=None,
                        help=f"Override DPO epochs (default from config = {DPO_EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=None,
                        help=f"Override DPO batch size (default from config = {BATCH_SIZE})")
    parser.add_argument("--lr-multiplier", type=float, default=None,
                        help=f"Override DPO learning rate multiplier (default from config = {LR_MULTIPLIER})")
    args = parser.parse_args()
    run_dpo(base_model=args.base_model, wait=args.wait, no_wandb=args.no_wandb,
            project=args.project, experiment=args.experiment,
            n_epochs=args.epochs, batch_size=args.batch_size, lr_multiplier=args.lr_multiplier)
