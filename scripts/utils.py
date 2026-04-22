import csv
import io
import os
import json
import time
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _wandb = None
    _WANDB_AVAILABLE = False


class WandbLogger:
    """Thin wrapper around wandb. Uses the system-wide login (wandb login).
    Pass disabled=True to opt out."""

    def __init__(self, project: str, experiment: str | None, config: dict, disabled: bool = False):
        self._run = None
        if disabled:
            return
        if not _WANDB_AVAILABLE:
            raise RuntimeError("wandb is not installed. Run: pip install wandb  (or pass --no-wandb)")
        self._run = _wandb.init(project=project, name=experiment, config=config)

    def define_sft_charts(self):
        if not self._run:
            return
        for metric in (
            "train/loss", "val/loss", "val/full_loss",
            "train/token_accuracy", "val/token_accuracy", "val/full_token_accuracy",
        ):
            _wandb.define_metric(metric, step_metric="azure_step")

    def define_dpo_charts(self):
        if not self._run:
            return
        for metric in (
            "train/loss", "val/loss", "val/full_loss",
            "train/token_accuracy", "val/token_accuracy", "val/full_token_accuracy",
            "train/reward_accuracy", "val/reward_accuracy",
            "train/reward_margin", "val/reward_margin",
        ):
            _wandb.define_metric(metric, step_metric="azure_step")

    def define_rft_charts(self):
        if not self._run:
            return
        for metric in (
            "train/loss", "val/loss",
            "train/reward", "val/reward",
            "train/kl_divergence",
        ):
            _wandb.define_metric(metric, step_metric="azure_step")

    def log(self, metrics: dict, step: int | None = None):
        if self._run:
            self._run.log(metrics, step=step)

    def finish(self):
        if self._run:
            self._run.finish()

    def __bool__(self):
        return self._run is not None

from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential
from azure.mgmt.cognitiveservices import CognitiveServicesManagementClient
from azure.mgmt.cognitiveservices.models import (
    Deployment, 
    DeploymentProperties, 
    DeploymentModel, 
    Sku
)


def get_openai_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=OPENAI_API_KEY,
        azure_endpoint=OPENAI_ENDPOINT,
        api_version="2025-04-01-preview",
    )


def get_credential() -> DefaultAzureCredential:
    return DefaultAzureCredential()


def get_cogsvc_client(credential: DefaultAzureCredential) -> CognitiveServicesManagementClient:
    return CognitiveServicesManagementClient(credential=credential, subscription_id=SUBSCRIPTION_ID)


def upload_file(client: AzureOpenAI, path: str, purpose: str = "fine-tune") -> str:
    print(f"Uploading {path}...")
    with open(path, "rb") as f:
        result = client.files.create(file=f, purpose=purpose)
    client.files.wait_for_processing(result.id)
    print(f"  -> File ID: {result.id}")
    return result.id


def wait_for_job(
    client: AzureOpenAI,
    job_id: str,
    poll_interval: int = 30,
    wandb_logger: "WandbLogger | None" = None,
) -> object:
    """Poll a fine-tuning job until it reaches a terminal state."""
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        print(f"  [{job_id}] status: {job.status}")
        if wandb_logger:
            wandb_logger.log({"job/status_code": _status_to_int(job.status)})
        if job.status in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(poll_interval)


def _status_to_int(status: str) -> int:
    return {"queued": 0, "running": 1, "succeeded": 2, "failed": -1, "cancelled": -2}.get(status, 0)


_SFT_COLUMN_MAP = {
    # shared (SFT + DPO)
    "train_loss": "train/loss",
    "valid_loss": "val/loss",
    "full_valid_loss": "val/full_loss",
    "train_mean_token_accuracy": "train/token_accuracy",
    "valid_mean_token_accuracy": "val/token_accuracy",
    "full_valid_mean_token_accuracy": "val/full_token_accuracy",
    # DPO-specific
    "train_reward_accuracy": "train/reward_accuracy",
    "valid_reward_accuracy": "val/reward_accuracy",
    "train_reward_margin": "train/reward_margin",
    "valid_reward_margin": "val/reward_margin",
    # RFT-specific
    "train_reward": "train/reward",
    "valid_reward": "val/reward",
    "train_kl": "train/kl_divergence",
    "train_kl_divergence": "train/kl_divergence",
}


def log_job_result(client: AzureOpenAI, job: object, wandb_logger: "WandbLogger") -> None:
    """Download result files from a completed job and log metrics to wandb."""
    if not wandb_logger:
        return
    result_files = getattr(job, "result_files", None) or []
    for file_id in result_files:
        try:
            content = client.files.content(file_id).read().decode("utf-8")
            reader = csv.DictReader(io.StringIO(content))
            for row in reader:
                step = int(row.get("step", 0))
                metrics = {}
                for k, v in row.items():
                    if k == "step" or v == "":
                        continue
                    try:
                        metrics[_SFT_COLUMN_MAP.get(k, k)] = float(v)
                    except ValueError:
                        pass
                if metrics:
                    metrics["azure_step"] = step
                    wandb_logger.log(metrics)
        except Exception as exc:
            print(f"  Warning: could not parse result file {file_id}: {exc}")

    summary = {}
    if getattr(job, "trained_tokens", None) is not None:
        summary["trained_tokens"] = job.trained_tokens
    if getattr(job, "fine_tuned_model", None):
        summary["fine_tuned_model"] = job.fine_tuned_model
    if summary:
        wandb_logger.log(summary)


def deploy_model(
    cogsvc_client: CognitiveServicesManagementClient,
    fine_tuned_model_id: str,
    deployment_name: str,
    capacity: int = 50,
) -> None:
    deployment_model = DeploymentModel(format="OpenAI", name=fine_tuned_model_id, version="1")
    deployment_properties = DeploymentProperties(model=deployment_model)
    deployment_sku = Sku(name="GlobalStandard", capacity=capacity)
    deployment_config = Deployment(properties=deployment_properties, sku=deployment_sku)

    print(f"Deploying {fine_tuned_model_id} as '{deployment_name}'...")
    poller = cogsvc_client.deployments.begin_create_or_update(
        resource_group_name=AZURE_RESOURCE_GROUP_NAME,
        account_name=RESOURCE_GROUP,
        deployment_name=deployment_name,
        deployment=deployment_config,
    )
    poller.result()
    print(f"  -> Deployment '{deployment_name}' ready.")


def delete_deployment(cogsvc_client: CognitiveServicesManagementClient, deployment_name: str) -> None:
    print(f"Deleting deployment: {deployment_name}")
    poller = cogsvc_client.deployments.begin_delete(
        resource_group_name=AZURE_RESOURCE_GROUP_NAME,
        account_name=RESOURCE_GROUP,
        deployment_name=deployment_name,
    )
    poller.result()
    print(f"  -> Deployment '{deployment_name}' deleted.")


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(records)} records -> {path}")
