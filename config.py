import os
from dotenv import load_dotenv

load_dotenv()

# Azure credentials
RESOURCE_GROUP = os.getenv("AZURE_AOAI_ACCOUNT")
OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
OPENAI_ENDPOINT = f"https://{RESOURCE_GROUP}.openai.azure.com"
SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID")
AZURE_RESOURCE_GROUP_NAME = "CIS-5270"

# Models
TEACHER_MODEL = "gpt-4.1-mini" # Generates LLM DPO examples and judge in LLM-as-judge
STUDENT_MODEL = "gpt-4.1-nano-2025-04-14"  # Fine-tuned model

# Dataset paths
DATA_DIR              = "data_files"
RAW_SOLUTIONS_FILE    = f"{DATA_DIR}/llm_solutions.jsonl"
SFT_TRAIN_FILE        = f"{DATA_DIR}/sft_train.jsonl"
SFT_VAL_FILE          = f"{DATA_DIR}/sft_val.jsonl"
DPO_TRAIN_FILE        = f"{DATA_DIR}/dpo_train.jsonl"
DPO_VAL_FILE          = f"{DATA_DIR}/dpo_val.jsonl"
RFT_TRAIN_FILE        = f"{DATA_DIR}/rft_train.jsonl"
RFT_VAL_FILE          = f"{DATA_DIR}/rft_val.jsonl"

# Reward weights (must sum to 1.0)
CORRECTNESS_WEIGHT = 0.7
STYLE_WEIGHT       = 0.3

# Style sub-weights (within style score)
STATIC_STYLE_WEIGHT = 0.6   # radon/AST metrics
JUDGE_STYLE_WEIGHT  = 0.4   # LLM-as-judge

# Training hyperparameters
SFT_EPOCHS        = 3
DPO_EPOCHS        = 2
RFT_EPOCHS        = 3
BATCH_SIZE        = 4
LR_MULTIPLIER     = 1.0

# Number of solutions to generate per problem
N_SOLUTIONS_PER_PROBLEM = 3  # mix of correct + incorrect
VAL_SPLIT = 0.1

# Weights & Biases
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "5270fp-finetuning")
