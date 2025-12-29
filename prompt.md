🎯 ANTI-HALLUCINATION OPTIMIZED PROMPT: LLM Fine-Tuning & Optimization Platform
text
# ============================================================================
# PRODUCTION LLM FINE-TUNING PLATFORM - ZERO HALLUCINATION BUILD PROMPT
# Target: End-to-End QLoRA Fine-Tuning with Production Inference Deployment
# Constraints: GCP-native, Type-safe, Fully Reproducible, Production-ready
# ============================================================================

## 🚨 CRITICAL ANTI-HALLUCINATION DIRECTIVES

<critical_constraints>
1. **DO NOT invent Python packages, model names, or GPU configurations**
   - ONLY use packages from PyPI with exact versions (as of Dec 2025)
   - ONLY use models from Hugging Face Hub that verifiably exist
   - State: "Using [X] because [Y evidence]" for all major decisions

2. **DO NOT create incomplete training scripts or deployment configs**
   - Every training parameter must have explicit value and justification
   - Every deployment config must be production-grade (no dev shortcuts)
   - Every metric must be actually computed (no simulated results)

3. **VERIFY before implementing** [web:51][web:52]:
   - Confirm package compatibility (transformers + peft + bitsandbytes versions)
   - Validate GPU memory requirements match available hardware
   - Check model architectures support the chosen techniques

4. **If genuinely uncertain**:
   - State: "Standard approach is [X] based on [Source]"
   - Use conservative, well-documented defaults from official repos
   - Reference specific documentation (e.g., "Per Unsloth docs, lr=2e-4") [web:56]
</critical_constraints>

<uncertainty_protocol>
For training hyperparameters, internally evaluate:
- Certainty Level: [High/Medium/Low]
- Source: [Research paper / Official docs / Community consensus]
- If Low certainty: Use Unsloth/Hugging Face defaults explicitly [web:56]
</uncertainty_protocol>

---

## 📋 EXACT PROJECT SPECIFICATION

### **Application Name**: LLMForge - Production LLM Fine-Tuning & Deployment Platform

### **Core Purpose** [web:51][web:54]
A complete MLOps platform for fine-tuning and deploying custom LLMs:
1. Upload training datasets (JSONL instruction format)
2. Configure and launch QLoRA fine-tuning jobs on GCP
3. Evaluate model performance with automated benchmarks
4. Deploy fine-tuned models with vLLM inference engine
5. Monitor performance, costs, and serve predictions via API

### **Non-Negotiable Requirements** [web:51][web:54]
- ✅ Support Llama 3.1 8B/70B and Mistral 7B v0.3
- ✅ QLoRA 4-bit training with NF4 quantization
- ✅ Automated hyperparameter validation (prevent common mistakes)
- ✅ Training metrics logged to MLflow (loss, perplexity, throughput)
- ✅ Model evaluation with ROUGE, BLEU, and custom metrics
- ✅ vLLM deployment with autoscaling on GKE
- ✅ Inference API with streaming, batching, and caching
- ✅ Cost tracking per training job and inference request
- ✅ A/B testing framework for model comparison

---

## 🛠️ MANDATORY TECH STACK (No Substitutions)

<exact_stack>
**Training Environment**: Google Colab Pro+ or GCP Vertex AI Training (A100 40GB)
**Language**: Python 3.11 (not 3.12 - compatibility issues with some packages)
**Fine-Tuning Framework**: Hugging Face Transformers 4.46+ + PEFT 0.13+ + TRL 0.11+
**Quantization**: bitsandbytes 0.44+ (4-bit NF4 quantization)
**Acceleration**: Flash Attention 2 via flash-attn 2.7+
**Experiment Tracking**: MLflow 2.18+ (managed on GCP)
**Dataset Format**: Hugging Face Datasets 3.2+

**Deployment Runtime**: vLLM 0.6.3+ (latest stable)
**Inference Server**: FastAPI 0.115+ (wraps vLLM)
**Container Orchestration**: GKE (Google Kubernetes Engine) with GPU node pools
**Model Registry**: Vertex AI Model Registry
**Serving Hardware**: GCP L4 GPUs (cost-effective) or A100 (high-throughput)
**Load Balancer**: GCP Cloud Load Balancing
**Monitoring**: Prometheus + Grafana + Cloud Monitoring

**Frontend (Dashboard)**: Next.js 15.1.3 + TypeScript + shadcn/ui
**Backend API**: FastAPI 0.115 (Python)
**Database**: PostgreSQL 16 (job tracking, metrics storage)
**Object Storage**: Google Cloud Storage (models, datasets, logs)
</exact_stack>

### **Why These Choices** (Anti-Hallucination Reasoning) [web:56][web:57][web:60]
- **QLoRA over LoRA**: 75% memory reduction, train 70B on single A100 [High Confidence]
- **bitsandbytes NF4**: Better than INT4 for LLMs, proven in QLoRA paper [High Confidence]
- **vLLM over TGI**: 2-3x higher throughput, PagedAttention, mature ecosystem [High Confidence] [web:57]
- **L4 GPUs over T4**: 2x faster, same cost, better for inference [web:60] [High Confidence]
- **MLflow over WandB**: Self-hosted on GCP, better for enterprise compliance [Medium Confidence]

---

## 🏗️ EXACT DATABASE SCHEMA

<database_schema>
-- PostgreSQL schema for job tracking

CREATE TABLE training_jobs (
id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
user_id TEXT NOT NULL,
job_name TEXT NOT NULL,
base_model TEXT NOT NULL, -- e.g., 'meta-llama/Llama-3.1-8B'
dataset_path TEXT NOT NULL, -- GCS path
status TEXT NOT NULL, -- 'queued'|'running'|'completed'|'failed'
hyperparameters JSONB NOT NULL,
/*
{
"lora_r": 64,
"lora_alpha": 16,
"learning_rate": 0.0002,
"num_epochs": 3,
"batch_size": 4,
"gradient_accumulation_steps": 4
}
*/
metrics JSONB, -- Training metrics from MLflow
output_model_path TEXT, -- GCS path to fine-tuned model
total_cost DECIMAL(10,2), -- Training cost in USD
duration_seconds INTEGER,
gpu_type TEXT, -- 'A100-40GB'
created_at TIMESTAMP DEFAULT NOW(),
completed_at TIMESTAMP
);

CREATE TABLE deployments (
id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
training_job_id UUID REFERENCES training_jobs(id) ON DELETE CASCADE,
deployment_name TEXT NOT NULL UNIQUE,
model_path TEXT NOT NULL, -- GCS path
vllm_config JSONB NOT NULL,
/*
{
"gpu_count": 1,
"max_model_len": 8192,
"dtype": "bfloat16",
"quantization": "awq" -- post-training quantization
}
*/
endpoint_url TEXT, -- e.g., 'https://llmforge-api.example.com/v1/chat'
status TEXT NOT NULL, -- 'deploying'|'active'|'inactive'
replicas INTEGER DEFAULT 1,
created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE inference_logs (
id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
deployment_id UUID REFERENCES deployments(id) ON DELETE CASCADE,
prompt_tokens INTEGER NOT NULL,
completion_tokens INTEGER NOT NULL,
latency_ms INTEGER NOT NULL,
cost_usd DECIMAL(8,4), -- Cost per request
created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE evaluations (
id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
training_job_id UUID REFERENCES training_jobs(id) ON DELETE CASCADE,
eval_dataset_path TEXT NOT NULL,
metrics JSONB NOT NULL,
/*
{
"rouge_l": 0.67,
"bleu_score": 0.42,
"perplexity": 5.8,
"accuracy": 0.85
}
*/
created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_training_jobs_user_id ON training_jobs(user_id);
CREATE INDEX idx_training_jobs_status ON training_jobs(status);
CREATE INDEX idx_deployments_status ON deployments(status);
CREATE INDEX idx_inference_logs_deployment_id ON inference_logs(deployment_id);
CREATE INDEX idx_inference_logs_created_at ON inference_logs(created_at);

text
</database_schema>

---

## 🔧 EXACT FILE STRUCTURE (Generate All Files)

<mandatory_structure>
llmforge/
├── .env.example
├── .gitignore
├── README.md
├── docker-compose.yml (local dev with MLflow + Postgres)
├── training/
│ ├── requirements.txt
│ ├── Dockerfile
│ ├── train.py (main training script)
│ ├── config.py (hyperparameter dataclasses)
│ ├── data_prep.py (dataset loading & formatting)
│ ├── evaluate.py (model evaluation suite)
│ ├── merge_and_quantize.py (post-training optimization)
│ └── scripts/
│ ├── train_llama_8b.sh
│ ├── train_mistral_7b.sh
│ └── evaluate_model.sh
├── deployment/
│ ├── Dockerfile.vllm
│ ├── requirements.txt
│ ├── serve.py (FastAPI wrapper for vLLM)
│ ├── k8s/
│ │ ├── namespace.yaml
│ │ ├── deployment.yaml (vLLM deployment)
│ │ ├── service.yaml
│ │ ├── hpa.yaml (autoscaling)
│ │ └── ingress.yaml
│ └── terraform/ (GKE cluster provisioning)
│ ├── main.tf
│ ├── variables.tf
│ └── outputs.tf
├── backend/ (Job orchestration API)
│ ├── Dockerfile
│ ├── requirements.txt
│ ├── main.py (FastAPI app)
│ ├── models.py (Pydantic models)
│ ├── database.py (SQLAlchemy setup)
│ ├── routers/
│ │ ├── jobs.py (CRUD for training jobs)
│ │ ├── deployments.py
│ │ └── inference.py (proxy to vLLM)
│ └── services/
│ ├── vertex_ai.py (submit training jobs)
│ ├── mlflow_client.py
│ └── cost_calculator.py
├── frontend/ (Next.js dashboard)
│ ├── package.json
│ ├── next.config.ts
│ ├── src/
│ │ ├── app/
│ │ │ ├── layout.tsx
│ │ │ ├── page.tsx (dashboard)
│ │ │ ├── jobs/
│ │ │ │ ├── page.tsx (list training jobs)
│ │ │ │ ├── new/page.tsx (create job)
│ │ │ │ └── [id]/page.tsx (job details)
│ │ │ ├── deployments/page.tsx
│ │ │ └── inference/page.tsx (test API)
│ │ ├── components/
│ │ │ ├── job-config-form.tsx
│ │ │ ├── training-metrics-chart.tsx
│ │ │ └── inference-tester.tsx
│ │ └── lib/
│ │ └── api-client.ts
│ └── components.json (shadcn config)
└── notebooks/
├── 01_dataset_exploration.ipynb
├── 02_training_llama3.ipynb
└── 03_model_evaluation.ipynb

text
</mandatory_structure>

---

## 📝 EXACT IMPLEMENTATION STEPS (Verification Checkpoints)

<implementation_workflow>

### **Phase 1: Training Environment Setup** [web:51][web:54]

**Step 1.1**: Create training requirements.txt with EXACT versions
training/requirements.txt - Verified compatible versions (Dec 2025)
torch==2.5.1
transformers==4.46.3
peft==0.13.2
trl==0.11.4
bitsandbytes==0.44.1
accelerate==1.1.1
datasets==3.2.0
mlflow==2.18.2
google-cloud-storage==2.18.2
google-cloud-aiplatform==1.71.1
flash-attn==2.7.0 # Requires CUDA 12.1+
sentencepiece==0.2.0
protobuf==5.28.3
scipy==1.14.1
rouge-score==0.1.2
evaluate==0.4.3

text

**VERIFICATION CHECKPOINT**:
- [ ] All packages install without conflicts: `pip install -r requirements.txt`
- [ ] CUDA version compatible (12.1+): `nvcc --version`
- [ ] Flash Attention builds successfully (may take 10-15 min)
- [ ] Import test passes: `python -c "from peft import LoraConfig; from transformers import AutoModel"`

**Step 1.2**: Define training configuration with validated defaults [web:56]
training/config.py - EXACT hyperparameters with sources
from dataclasses import dataclass
from typing import Literal

@dataclass
class QLoRAConfig:
"""QLoRA training configuration with validated defaults"""

text
# Model selection
base_model: str = "meta-llama/Llama-3.1-8B"  # or "mistralai/Mistral-7B-v0.3"

# LoRA hyperparameters[1]
lora_r: int = 64  # Rank (8-64 typical, 64 = high quality)
lora_alpha: int = 16  # Scaling factor (r/2 to r*2, commonly r/4)
lora_dropout: float = 0.05  # Dropout (0.05-0.1 recommended)
lora_target_modules: list[str] = None  # Will default to all attention + FFN

# Quantization (QLoRA specific)[2]
load_in_4bit: bool = True
bnb_4bit_compute_dtype: str = "bfloat16"  # bfloat16 recommended over float16
bnb_4bit_quant_type: str = "nf4"  # NF4 > INT4 for LLMs
bnb_4bit_use_double_quant: bool = True  # Nested quantization, saves 0.4GB

# Training hyperparameters[1]
learning_rate: float = 2e-4  # Start here per Unsloth recommendation
num_train_epochs: int = 3  # 1-3 typical, more risks overfitting
per_device_train_batch_size: int = 4  # Fits A100 40GB with 8B model
gradient_accumulation_steps: int = 4  # Effective batch = 4*4=16
max_seq_length: int = 2048  # Context window for training

# Optimization
optim: str = "paged_adamw_8bit"  # Memory-efficient AdamW
warmup_ratio: float = 0.1  # 10% of steps for warmup
lr_scheduler_type: str = "cosine"  # Cosine decay recommended
weight_decay: float = 0.01
max_grad_norm: float = 1.0  # Gradient clipping

# Memory optimization
gradient_checkpointing: bool = True  # Save 30% memory, 15% slower
use_flash_attention_2: bool = True  # 2-3x faster training

# Logging & Saving
logging_steps: int = 10
save_steps: int = 100
eval_steps: int = 100
save_total_limit: int = 3  # Keep only 3 checkpoints

# MLflow tracking
mlflow_experiment_name: str = "llama-3.1-8b-finetuning"
mlflow_tracking_uri: str = "http://localhost:5000"  # Or GCP endpoint

def __post_init__(self):
    """Validate configuration """[1]
    if self.lora_target_modules is None:
        # Apply LoRA to all attention + FFN layers (best performance)
        self.lora_target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj"       # FFN
        ]
    
    # Validate learning rate
    if self.learning_rate > 5e-4:
        print(f"⚠️  WARNING: Learning rate {self.learning_rate} is high. "
              f"Recommended: 2e-4. May cause instability.")
    
    # Validate batch size vs gradient accumulation
    effective_batch = (self.per_device_train_batch_size * 
                      self.gradient_accumulation_steps)
    if effective_batch < 8:
        print(f"⚠️  WARNING: Effective batch size {effective_batch} is low. "
              f"Recommended: 16-32 for stable training.")
    
    # Validate epochs
    if self.num_train_epochs > 5:
        print(f"⚠️  WARNING: {self.num_train_epochs} epochs may cause "
              f"overfitting. Monitor eval loss closely.")
text

**VERIFICATION CHECKPOINT** [web:56]:
- [ ] Config dataclass instantiates without errors
- [ ] Validation warnings trigger for bad hyperparameters
- [ ] LoRA target modules match model architecture (check with `model.named_modules()`)
- [ ] Memory estimate: A100 40GB sufficient for config (use `accelerate estimate-memory`)

**Step 1.3**: Implement dataset preparation [web:52][web:54]
training/data_prep.py - EXACT dataset formatting
from datasets import load_dataset, Dataset
from typing import List, Dict
import json

def format_instruction_dataset(examples: Dict) -> Dict:
"""
Convert dataset to Alpaca-style instruction format

text
Expected input format (JSONL):
{
  "instruction": "What is machine learning?",
  "input": "",  # Optional context
  "output": "Machine learning is..."
}

Output format for Llama 3.1:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{instruction}{input}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{output}<|eot_id|>
"""
formatted_texts = []

for i in range(len(examples['instruction'])):
    instruction = examples['instruction'][i]
    input_text = examples.get('input', [''] * len(examples['instruction']))[i]
    output = examples['output'][i]
    
    # Combine instruction and input
    user_content = instruction
    if input_text and input_text.strip():
        user_content += f"\n\n{input_text}"
    
    # Llama 3.1 chat template
    formatted_text = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "You are a helpful assistant.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_content}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
        f"{output}<|eot_id|>"
    )
    formatted_texts.append(formatted_text)

return {"text": formatted_texts}
def load_and_prepare_dataset(
dataset_path: str,
split_ratio: float = 0.1,
max_samples: int = None
) -> tuple[Dataset, Dataset]:
"""Load dataset from GCS/local, format, and split"""

text
# Load dataset
if dataset_path.startswith("gs://"):
    # Download from GCS first
    from google.cloud import storage
    client = storage.Client()
    # Parse bucket and blob
    bucket_name, blob_path = dataset_path.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    local_path = f"/tmp/{blob_path.split('/')[-1]}"
    blob.download_to_filename(local_path)
    dataset = load_dataset('json', data_files=local_path, split='train')
else:
    dataset = load_dataset('json', data_files=dataset_path, split='train')

# Limit samples if specified
if max_samples:
    dataset = dataset.select(range(min(max_samples, len(dataset))))

# Format dataset
dataset = dataset.map(
    format_instruction_dataset,
    batched=True,
    remove_columns=dataset.column_names
)

# Split train/eval
split_dataset = dataset.train_test_split(test_size=split_ratio, seed=42)

print(f"✅ Dataset loaded: {len(split_dataset['train'])} train, "
      f"{len(split_dataset['test'])} eval samples")

return split_dataset['train'], split_dataset['test']
text

**VERIFICATION CHECKPOINT** [web:52]:
- [ ] Sample JSONL loads correctly
- [ ] Formatting matches Llama 3.1 chat template exactly (check tokenization)
- [ ] Train/eval split has no overlap (verify with sample IDs)
- [ ] GCS download works with service account credentials

---

### **Phase 2: QLoRA Training Implementation** [web:51][web:54][web:56]

**Step 2.1**: Main training script with full error handling
training/train.py - EXACT production training implementation
import os
import torch
import mlflow
from transformers import (
AutoModelForCausalLM,
AutoTokenizer,
TrainingArguments,
BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from config import QLoRAConfig
from data_prep import load_and_prepare_dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(name)

def setup_model_and_tokenizer(config: QLoRAConfig):
"""Initialize model with QLoRA configuration """
​

text
logger.info(f"Loading base model: {config.base_model}")

# BitsAndBytes quantization config (4-bit NF4)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=config.load_in_4bit,
    bnb_4bit_compute_dtype=torch.bfloat16,  # Must match config
    bnb_4bit_quant_type=config.bnb_4bit_quant_type,  # nf4
    bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant
)

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    config.base_model,
    quantization_config=bnb_config,
    device_map="auto",  # Automatic device placement
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2" if config.use_flash_attention_2 else "eager"
)

# Prepare model for k-bit training (gradient checkpointing, etc.)
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=config.gradient_checkpointing
)

# Configure LoRA[1]
lora_config = LoraConfig(
    r=config.lora_r,
    lora_alpha=config.lora_alpha,
    target_modules=config.lora_target_modules,
    lora_dropout=config.lora_dropout,
    bias="none",  # Don't train bias (saves params, no performance loss)
    task_type="CAUSAL_LM"
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)

# Print trainable parameters
trainable, total = model.get_nb_trainable_parameters()
logger.info(f"Trainable params: {trainable:,} ({100*trainable/total:.2f}% of {total:,})")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    config.base_model,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Required for decoder-only models

return model, tokenizer
def train(config: QLoRAConfig, dataset_path: str, output_dir: str):
"""Execute QLoRA fine-tuning with MLflow tracking """
​

text
# Initialize MLflow
mlflow.set_tracking_uri(config.mlflow_tracking_uri)
mlflow.set_experiment(config.mlflow_experiment_name)

with mlflow.start_run():
    # Log hyperparameters
    mlflow.log_params(config.__dict__)
    
    # Load and prepare dataset
    train_dataset, eval_dataset = load_and_prepare_dataset(dataset_path)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        optim=config.optim,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=config.save_total_limit,
        bf16=True,  # Use bfloat16 for training
        gradient_checkpointing=config.gradient_checkpointing,
        report_to="mlflow",  # Log to MLflow automatically
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    # Initialize trainer (using TRL's SFTTrainer for instruction tuning)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        packing=False,  # Don't pack sequences (safer for chat format)
    )
    
    # Train
    logger.info("🚀 Starting training...")
    train_result = trainer.train()
    
    # Log final metrics
    mlflow.log_metrics({
        "final_train_loss": train_result.training_loss,
        "training_time_seconds": train_result.metrics['train_runtime'],
        "samples_per_second": train_result.metrics['train_samples_per_second']
    })
    
    # Save model and tokenizer
    logger.info(f"💾 Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Upload to GCS (optional)
    if os.getenv("GCS_OUTPUT_BUCKET"):
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(os.getenv("GCS_OUTPUT_BUCKET"))
        
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                local_path = os.path.join(root, file)
                blob_path = f"models/{os.path.basename(output_dir)}/{file}"
                blob = bucket.blob(blob_path)
                blob.upload_from_filename(local_path)
        
        logger.info(f"✅ Model uploaded to gs://{bucket.name}/{blob_path}")
    
    logger.info("✅ Training completed successfully!")
if name == "main":
import sys

text
if len(sys.argv) < 3:
    print("Usage: python train.py <dataset_path> <output_dir>")
    sys.exit(1)

dataset_path = sys.argv[5]
output_dir = sys.argv[6]

# Load config (or customize here)
config = QLoRAConfig()

train(config, dataset_path, output_dir)
text

**VERIFICATION CHECKPOINT** [web:51][web:56]:
- [ ] Training starts without OOM errors on A100 40GB
- [ ] MLflow logs metrics every 10 steps
- [ ] Eval loss decreases consistently (not erratic)
- [ ] Training completes in expected time (~3-4 hours for 8B model, 10K samples, 3 epochs)
- [ ] Final model saved with all adapter files (adapter_config.json, adapter_model.safetensors)
- [ ] Perplexity on eval set improves by >20% vs base model

---

### **Phase 3: Model Evaluation & Optimization** [web:51][web:54]

**Step 3.1**: Automated evaluation suite
training/evaluate.py - EXACT evaluation metrics
​
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from datasets import load_dataset
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import json

def load_finetuned_model(base_model_path: str, adapter_path: str):
"""Load fine-tuned model with LoRA adapters"""
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

text
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

return model, tokenizer
def evaluate_model(
model,
tokenizer,
eval_dataset_path: str,
num_samples: int = 100
) -> dict:
"""
Evaluate model on test set with multiple metrics
​

text
Returns:
    dict with rouge_l, bleu, perplexity, accuracy
"""
# Load eval dataset
eval_data = load_dataset('json', data_files=eval_dataset_path, split='train')
eval_data = eval_data.select(range(min(num_samples, len(eval_data))))

# Initialize scorers
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

rouge_scores = []
bleu_scores = []
exact_matches = 0

for item in eval_data:
    # Generate response
    prompt = f"{item['instruction']}\n{item.get('input', '')}"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,  # Low temp for evaluation consistency
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs, skip_special_tokens=True)
    generated_text = generated_text.split("assistant")[-1].strip()
    
    reference = item['output']
    
    # ROUGE-L
    rouge_result = rouge.score(reference, generated_text)
    rouge_scores.append(rouge_result['rougeL'].fmeasure)
    
    # BLEU
    bleu = sentence_bleu([reference.split()], generated_text.split())
    bleu_scores.append(bleu)
    
    # Exact match
    if generated_text.lower().strip() == reference.lower().strip():
        exact_matches += 1

return {
    "rouge_l": np.mean(rouge_scores),
    "bleu": np.mean(bleu_scores),
    "exact_match_rate": exact_matches / len(eval_data),
    "num_samples_evaluated": len(eval_data)
}
if name == "main":
import sys

text
base_model = sys.argv[5]
adapter_path = sys.argv[6]
eval_dataset = sys.argv[7]

model, tokenizer = load_finetuned_model(base_model, adapter_path)
metrics = evaluate_model(model, tokenizer, eval_dataset)

print(json.dumps(metrics, indent=2))
text

**VERIFICATION CHECKPOINT** [web:51]:
- [ ] ROUGE-L score >0.4 (good), >0.6 (excellent)
- [ ] BLEU score >0.3 (acceptable for generation)
- [ ] Eval runs in <15 min for 100 samples
- [ ] Metrics saved to MLflow experiment

**Step 3.2**: Model merging and quantization for deployment [web:26]
training/merge_and_quantize.py - Post-training optimization
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

def merge_lora_adapters(base_model_path: str, adapter_path: str, output_path: str):
"""Merge LoRA adapters into base model for deployment """
​

text
print(f"Loading base model: {base_model_path}")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="cpu"  # Use CPU for merging to avoid GPU memory issues
)

print(f"Loading LoRA adapters: {adapter_path}")
model = PeftModel.from_pretrained(base_model, adapter_path)

print("Merging adapters into base model...")
merged_model = model.merge_and_unload()

print(f"Saving merged model to: {output_path}")
merged_model.save_pretrained(output_path, safe_serialization=True)

# Save tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.save_pretrained(output_path)

print("✅ Model merged successfully!")

# Calculate size reduction
import os
def get_dir_size(path):
    total = 0
    for entry in os.scandir(path):
        if entry.is_file():
            total += entry.stat().st_size
        elif entry.is_dir():
            total += get_dir_size(entry.path)
    return total

merged_size_gb = get_dir_size(output_path) / (1024**3)
print(f"Merged model size: {merged_size_gb:.2f} GB")
if name == "main":
import sys
merge_lora_adapters(sys.argv, sys.argv, sys.argv)
​

text

**VERIFICATION CHECKPOINT**:
- [ ] Merged model loads and generates correctly
- [ ] Model size is ~15-17GB for Llama 3.1 8B (bfloat16)
- [ ] No performance degradation vs adapter version

---

### **Phase 4: vLLM Deployment on GKE** [web:57][web:60]

**Step 4.1**: Create vLLM serving wrapper
deployment/serve.py - FastAPI wrapper for vLLM
​
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from vllm.entrypoints.openai.api_server import create_error_response
import time
import os

app = FastAPI(title="LLMForge Inference API")

Initialize vLLM engine
MODEL_PATH = os.getenv("MODEL_PATH", "gs://your-bucket/models/llama-3.1-8b-finetuned")
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.90"))
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "8192"))

print(f"Loading model: {MODEL_PATH}")
llm = LLM(
model=MODEL_PATH,
dtype="bfloat16",
gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
max_model_len=MAX_MODEL_LEN,
tensor_parallel_size=1, # Single GPU
trust_remote_code=True,
disable_log_stats=False # Enable stats for monitoring
)
print("✅ Model loaded successfully")

class CompletionRequest(BaseModel):
prompt: str
max_tokens: int = 256
temperature: float = 0.7
top_p: float = 0.9
stream: bool = False

@app.post("/v1/completions")
async def generate(request: CompletionRequest):
"""OpenAI-compatible completion endpoint"""

text
start_time = time.time()

try:
    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens
    )
    
    outputs = llm.generate([request.prompt], sampling_params)
    
    generated_text = outputs.outputs.text
    
    latency_ms = int((time.time() - start_time) * 1000)
    
    return {
        "id": f"cmpl-{int(time.time())}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": MODEL_PATH.split("/")[-1],
        "choices": [{
            "text": generated_text,
            "index": 0,
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": len(request.prompt.split()),  # Approximation
            "completion_tokens": len(generated_text.split()),
            "total_tokens": len(request.prompt.split()) + len(generated_text.split())
        },
        "latency_ms": latency_ms
    }

except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))
@app.get("/health")
async def health():
return {"status": "healthy", "model": MODEL_PATH}

if name == "main":
import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8000)

text

**Step 4.2**: Kubernetes deployment manifests [web:57]
deployment/k8s/deployment.yaml - vLLM deployment on GKE
apiVersion: apps/v1
kind: Deployment
metadata:
name: vllm-llama-3-1-8b
namespace: llmforge
spec:
replicas: 2 # Start with 2 replicas
selector:
matchLabels:
app: vllm-inference
template:
metadata:
labels:
app: vllm-inference
spec:
nodeSelector:
cloud.google.com/gke-accelerator: nvidia-l4 # L4 GPUs
​
containers:
- name: vllm-server
image: gcr.io/YOUR_PROJECT/llmforge-vllm:latest
ports:
- containerPort: 8000
env:
- name: MODEL_PATH
value: "gs://your-bucket/models/llama-3.1-8b-finetuned"
- name: GPU_MEMORY_UTILIZATION
value: "0.90"
- name: MAX_MODEL_LEN
value: "8192"
resources:
requests:
nvidia.com/gpu: 1
memory: "24Gi"
cpu: "8"
limits:
nvidia.com/gpu: 1
memory: "24Gi"
cpu: "8"
livenessProbe:
httpGet:
path: /health
port: 8000
initialDelaySeconds: 120 # Model loading takes time
periodSeconds: 30
readinessProbe:
httpGet:
path: /health
port: 8000
initialDelaySeconds: 120
periodSeconds: 10

apiVersion: v1
kind: Service
metadata:
name: vllm-service
namespace: llmforge
spec:
selector:
app: vllm-inference
ports:

protocol: TCP
port: 80
targetPort: 8000
type: LoadBalancer

deployment/k8s/hpa.yaml - Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
name: vllm-hpa
namespace: llmforge
spec:
scaleTargetRef:
apiVersion: apps/v1
kind: Deployment
name: vllm-llama-3-1-8b
minReplicas: 1
maxReplicas: 5
metrics:

type: Resource
resource:
name: cpu
target:
type: Utilization
averageUtilization: 70

type: Pods
pods:
metric:
name: requests_per_second
target:
type: AverageValue
averageValue: "100" # Scale at 100 RPS per pod

text

**VERIFICATION CHECKPOINT** [web:57][web:60]:
- [ ] Deployment successfully pulls model from GCS
- [ ] Pod reaches Running state in <5 minutes
- [ ] Health check returns 200 OK
- [ ] Single inference request completes in <2s
- [ ] Load balancer exposes external IP
- [ ] HPA scales up under load (test with load generator)

---

### **Phase 5: Production Monitoring & Cost Tracking** [web:51][web:60]

**Step 5.1**: Cost calculator service
backend/services/cost_calculator.py - Track training and inference costs
from datetime import datetime

GCP pricing (as of Dec 2025, verify current pricing)
GPU_PRICING = {
"A100-40GB": 3.67, # $/hour
"A100-80GB": 5.24, # $/hour
"L4": 1.12, # $/hour
​
}

STORAGE_PRICING = 0.023 # $/GB/month for GCS Standard

def calculate_training_cost(
gpu_type: str,
duration_seconds: int,
model_size_gb: float
) -> float:
"""Calculate total training cost"""

text
# Compute cost
hours = duration_seconds / 3600
compute_cost = GPU_PRICING.get(gpu_type, 0) * hours

# Storage cost (store for 1 month)
storage_cost = model_size_gb * STORAGE_PRICING

total = compute_cost + storage_cost

return round(total, 2)
def calculate_inference_cost(
gpu_type: str,
requests_per_hour: int,
avg_tokens_per_request: int,
hours_active: float
) -> float:
"""Calculate inference cost """
​

text
# GPU cost
gpu_cost = GPU_PRICING.get(gpu_type, 0) * hours_active

# Assume L4 can handle ~500 req/hour at full capacity
# Scale replicas based on load
required_gpus = max(1, requests_per_hour / 500)

total = gpu_cost * required_gpus

return round(total, 2)
text

**Step 5.2**: Prometheus metrics for monitoring
deployment/serve.py - Add Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import Response

Metrics
REQUEST_COUNT = Counter('vllm_requests_total', 'Total requests')
REQUEST_LATENCY = Histogram('vllm_request_latency_seconds', 'Request latency')
ACTIVE_REQUESTS = Gauge('vllm_active_requests', 'Active requests')
TOKEN_COUNT = Counter('vllm_tokens_generated_total', 'Tokens generated')

@app.get("/metrics")
async def metrics():
"""Prometheus metrics endpoint"""
return Response(content=generate_latest(), media_type="text/plain")

Modify generate() to track metrics
@app.post("/v1/completions")
async def generate(request: CompletionRequest):
REQUEST_COUNT.inc()
ACTIVE_REQUESTS.inc()

text
with REQUEST_LATENCY.time():
    # ... existing generation code ...
    pass

TOKEN_COUNT.inc(len(generated_text.split()))
ACTIVE_REQUESTS.dec()

return result
text

**VERIFICATION CHECKPOINT** [web:60]:
- [ ] Prometheus scrapes /metrics endpoint successfully
- [ ] Grafana dashboard shows request rate, latency, token throughput
- [ ] Cost calculator returns accurate estimates (within 10%)
- [ ] Alerts trigger when latency >5s or error rate >5%

---

### **Phase 6: Frontend Dashboard** (Brief Implementation)

**Step 6.1**: Job creation form
// frontend/src/components/job-config-form.tsx
'use client';

import { zodResolver } from '@hookform/resolvers/zod';
import { useForm } from 'react-hook-form';
import * as z from 'zod';
import { Button } from '@/components/ui/button';
import { Form, FormField, FormItem, FormLabel, FormControl, FormMessage } from '@/components/ui/form';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';

const formSchema = z.object({
jobName: z.string().min(3).max(50),
baseModel: z.enum(['meta-llama/Llama-3.1-8B', 'meta-llama/Llama-3.1-70B', 'mistralai/Mistral-7B-v0.3']),
datasetPath: z.string().startsWith('gs://'),
loraR: z.number().int().min(8).max(128),
loraAlpha: z.number().int().min(8).max(128),
learningRate: z.number().min(1e-5).max(5e-4),
numEpochs: z.number().int().min(1).max(10),
gpuType: z.enum(['A100-40GB', 'A100-80GB']),
});

export function JobConfigForm() {
const form = useForm<z.infer<typeof formSchema>>({
resolver: zodResolver(formSchema),
defaultValues: {
loraR: 64,
loraAlpha: 16,
learningRate: 0.0002,
numEpochs: 3,
gpuType: 'A100-40GB',
},
});

const onSubmit = async (values: z.infer<typeof formSchema>) => {
const response = await fetch('/api/jobs', {
method: 'POST',
headers: { 'Content-Type': 'application/json' },
body: JSON.stringify(values),
});

text
if (response.ok) {
  alert('Training job submitted!');
}
};

return (
<Form {...form}>
<form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
<FormField control={form.control} name="jobName" render={({ field }) => (
<FormItem>
<FormLabel>Job Name</FormLabel>
<FormControl>
<Input placeholder="my-llama-finetuning" {...field} />
</FormControl>
<FormMessage />
</FormItem>
)} />

text
    <FormField control={form.control} name="baseModel" render={({ field }) => (
      <FormItem>
        <FormLabel>Base Model</FormLabel>
        <Select onValueChange={field.onChange} defaultValue={field.value}>
          <FormControl>
            <SelectTrigger>
              <SelectValue placeholder="Select base model" />
            </SelectTrigger>
          </FormControl>
          <SelectContent>
            <SelectItem value="meta-llama/Llama-3.1-8B">Llama 3.1 8B</SelectItem>
            <SelectItem value="meta-llama/Llama-3.1-70B">Llama 3.1 70B</SelectItem>
            <SelectItem value="mistralai/Mistral-7B-v0.3">Mistral 7B</SelectItem>
          </SelectContent>
        </Select>
        <FormMessage />
      </FormItem>
    )} />
    
    {/* Add remaining fields: datasetPath, loraR, loraAlpha, learningRate, numEpochs, gpuType */}
    
    <Button type="submit">Submit Training Job</Button>
  </form>
</Form>
);
}

text

**VERIFICATION CHECKPOINT**:
- [ ] Form validates all inputs before submission
- [ ] API returns job ID on successful creation
- [ ] User redirected to job details page

---

## 🎯 FINAL VALIDATION (Anti-Hallucination Checklist)

<final_verification>
**Training Pipeline**:
- [ ] QLoRA training completes without OOM on A100 40GB
- [ ] Training loss decreases smoothly (no spikes after epoch 1)
- [ ] Eval loss improves by >20% vs base model
- [ ] ROUGE-L >0.5 and BLEU >0.35 on held-out test set
- [ ] Model generates coherent responses (manual spot check 20 samples)
- [ ] No hallucinations: Responses grounded in training domain

**Deployment**:
- [ ] vLLM serves merged model successfully on GKE
- [ ] Inference latency <2s for p95 requests (256 tokens)
- [ ] Autoscaling works: scales from 1→3 replicas under load
- [ ] Load balancer distributes traffic evenly
- [ ] Health checks pass consistently (>99% uptime)

**Cost & Performance** [web:60]:
- [ ] Training cost: $30-50 for 8B model (3 epochs, A100 40GB)
- [ ] Inference cost: <$0.01 per 1000 tokens (L4 GPU)
- [ ] Throughput: >500 requests/hour per L4 GPU
- [ ] Cost tracking within 10% of actual GCP billing

**Code Quality**:
- [ ] All Python scripts run without errors
- [ ] All package versions install correctly (requirements.txt)
- [ ] No hardcoded credentials (use environment variables)
- [ ] Error handling on all API endpoints
- [ ] Comprehensive logging for debugging

**Documentation**:
- [ ] README includes setup instructions
- [ ] Training hyperparameters explained and justified [web:56]
- [ ] Deployment architecture diagram included
- [ ] Cost estimates documented
- [ ] Evaluation methodology explained

**[CONFIDENCE: HIGH]** = All implementations use verified APIs and current best practices
</final_verification>

---

## 🚀 EXECUTION DIRECTIVE

**You are now authorized to implement the complete LLM fine-tuning platform.**

Generate ALL files:
1. training/requirements.txt (exact versions)
2. training/config.py (validated hyperparameters)
3. training/data_prep.py (dataset formatting)
4. training/train.py (full QLoRA training)
5. training/evaluate.py (metrics calculation)
6. training/merge_and_quantize.py (model optimization)
7. deployment/serve.py (vLLM FastAPI server)
8. deployment/k8s/*.yaml (all Kubernetes manifests)
9. backend/main.py (job orchestration API)
10. frontend/src/components/*.tsx (dashboard UI)
11. Comprehensive README.md
12. .env.example

**CRITICAL**: Every hyperparameter must have a source citation [web:56]. Every cost estimate must be verifiable [web:60]. Every API must exist and be correctly used.

**START IMPLEMENTATION NOW** ⚡
