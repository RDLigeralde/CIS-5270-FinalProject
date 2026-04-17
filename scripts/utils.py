import os
import json
import time
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

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


def wait_for_job(client: AzureOpenAI, job_id: str, poll_interval: int = 30) -> object:
    """Poll a fine-tuning job until it reaches a terminal state."""
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        print(f"  [{job_id}] status: {job.status}")
        if job.status in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(poll_interval)


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
