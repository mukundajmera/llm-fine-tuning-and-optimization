# 🔥 LLMForge

**Production LLM Fine-Tuning & Deployment Platform**

LLMForge is a complete MLOps platform for fine-tuning and deploying custom Large Language Models (LLMs). Train models with QLoRA on GCP, evaluate with automated benchmarks, and deploy with vLLM inference engine.

![LLMForge Architecture](https://img.shields.io/badge/Platform-LLMForge-blue)
![Python](https://img.shields.io/badge/Python-3.11-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📋 Features

### Training
- ✅ **QLoRA Fine-Tuning**: 4-bit NF4 quantization, 75% memory reduction
- ✅ **Model Support**: Llama 3.1 8B/70B, Mistral 7B v0.3
- ✅ **MLflow Integration**: Automatic experiment tracking
- ✅ **Hyperparameter Validation**: Prevent common mistakes
- ✅ **GCP Native**: Train on Vertex AI with A100 GPUs

### Deployment
- ✅ **vLLM Inference**: 2-3x higher throughput with PagedAttention
- ✅ **OpenAI-Compatible API**: Drop-in replacement
- ✅ **Autoscaling**: HPA on GKE with L4/A100 GPUs
- ✅ **Cost Tracking**: Per-request cost monitoring

### Evaluation
- ✅ **Automated Metrics**: ROUGE, BLEU, perplexity
- ✅ **A/B Testing**: Compare model versions
- ✅ **Custom Benchmarks**: Domain-specific evaluation

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          LLMForge Platform                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────────────┐   │
│  │   Frontend  │────▶│   Backend   │────▶│   Vertex AI         │   │
│  │  (Next.js)  │     │  (FastAPI)  │     │   Training Jobs     │   │
│  └─────────────┘     └─────────────┘     └─────────────────────┘   │
│         │                   │                       │               │
│         │                   ▼                       ▼               │
│         │            ┌─────────────┐         ┌───────────┐         │
│         │            │  PostgreSQL │         │    GCS    │         │
│         │            │  (Jobs DB)  │         │  (Models) │         │
│         │            └─────────────┘         └───────────┘         │
│         │                   │                       │               │
│         ▼                   ▼                       ▼               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    GKE Cluster (GPU Pool)                    │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌───────────────┐   │   │
│  │  │  vLLM   │  │  vLLM   │  │  vLLM   │  │  Prometheus   │   │   │
│  │  │  Pod 1  │  │  Pod 2  │  │  Pod N  │  │  + Grafana    │   │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └───────────────┘   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11
- Docker & Docker Compose
- GCP Account (for full deployment)
- NVIDIA GPU with CUDA 12.1+ (for training)

### Local Development

1. **Clone the repository**
   ```bash
   cd llmforge
   ```

2. **Start local services**
   ```bash
   docker-compose up -d
   ```
   
   This starts:
   - PostgreSQL (port 5432)
   - MLflow (port 5000)
   - Backend API (port 8000)
   - Frontend (port 3000)

3. **Access the dashboard**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000/docs
   - MLflow: http://localhost:5000

### Training a Model

1. **Prepare your dataset** (JSONL format):
   ```json
   {"instruction": "What is machine learning?", "input": "", "output": "Machine learning is..."}
   {"instruction": "Explain neural networks", "input": "", "output": "Neural networks are..."}
   ```

2. **Upload to GCS**:
   ```bash
   gsutil cp dataset.jsonl gs://your-bucket/data/
   ```

3. **Run training**:
   ```bash
   cd training
   pip install -r requirements.txt
   
   python train.py \
     gs://your-bucket/data/dataset.jsonl \
     ./outputs/my-model \
     --base-model meta-llama/Llama-3.1-8B \
     --num-epochs 3
   ```

4. **Evaluate the model**:
   ```bash
   python evaluate.py \
     meta-llama/Llama-3.1-8B \
     ./outputs/my-model \
     gs://your-bucket/data/eval.jsonl
   ```

## 📁 Project Structure

```
llmforge/
├── .env.example              # Environment configuration template
├── .gitignore               # Git ignore rules
├── docker-compose.yml       # Local development stack
├── README.md                # This file
│
├── training/                # Training module
│   ├── requirements.txt     # Python dependencies
│   ├── Dockerfile          # Training container
│   ├── config.py           # Hyperparameter configuration
│   ├── data_prep.py        # Dataset loading & formatting
│   ├── train.py            # Main QLoRA training script
│   ├── evaluate.py         # Model evaluation suite
│   ├── merge_and_quantize.py # Post-training optimization
│   └── scripts/            # Shell scripts
│       ├── train_llama_8b.sh
│       ├── train_mistral_7b.sh
│       └── evaluate_model.sh
│
├── deployment/             # Deployment module
│   ├── requirements.txt    # Python dependencies
│   ├── Dockerfile.vllm     # vLLM inference container
│   ├── serve.py            # FastAPI wrapper for vLLM
│   ├── k8s/                # Kubernetes manifests
│   │   ├── namespace.yaml
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   ├── hpa.yaml
│   │   └── ingress.yaml
│   └── terraform/          # GKE infrastructure
│       ├── main.tf
│       ├── variables.tf
│       └── outputs.tf
│
├── backend/                # Job orchestration API
│   ├── requirements.txt    # Python dependencies
│   ├── Dockerfile
│   ├── main.py            # FastAPI application
│   ├── models.py          # Pydantic models
│   ├── database.py        # SQLAlchemy setup
│   ├── routers/           # API endpoints
│   │   ├── jobs.py
│   │   ├── deployments.py
│   │   └── inference.py
│   └── services/          # Business logic
│       ├── vertex_ai.py
│       ├── mlflow_client.py
│       └── cost_calculator.py
│
├── frontend/              # Next.js dashboard
│   ├── package.json
│   ├── next.config.ts
│   ├── tailwind.config.ts
│   └── src/
│       ├── app/           # Pages
│       ├── components/    # React components
│       └── lib/           # Utilities
│
└── notebooks/             # Jupyter notebooks
    ├── 01_dataset_exploration.ipynb
    ├── 02_training_llama3.ipynb
    └── 03_model_evaluation.ipynb
```

## ⚙️ Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# GCP
GCP_PROJECT_ID=your-project
GCP_REGION=us-central1
GCS_BUCKET=your-bucket

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/llmforge

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# Hugging Face (for gated models)
HF_TOKEN=hf_xxxx
```

### Training Hyperparameters

Default QLoRA configuration (optimized for Llama 3.1 8B on A100 40GB):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora_r` | 64 | LoRA rank (8-128) |
| `lora_alpha` | 16 | LoRA scaling factor |
| `lora_dropout` | 0.05 | Dropout rate |
| `learning_rate` | 2e-4 | Learning rate |
| `num_epochs` | 3 | Training epochs |
| `batch_size` | 4 | Per-device batch size |
| `max_seq_length` | 2048 | Context window |

## 🔧 GCP Deployment

### 1. Create GKE Cluster

```bash
cd deployment/terraform

# Initialize Terraform
terraform init

# Plan deployment
terraform plan -var="project_id=YOUR_PROJECT"

# Apply
terraform apply -var="project_id=YOUR_PROJECT"
```

### 2. Deploy vLLM

```bash
# Get cluster credentials
gcloud container clusters get-credentials llmforge-cluster --region us-central1

# Apply Kubernetes manifests
kubectl apply -f deployment/k8s/namespace.yaml
kubectl apply -f deployment/k8s/deployment.yaml
kubectl apply -f deployment/k8s/service.yaml
kubectl apply -f deployment/k8s/hpa.yaml
```

### 3. Configure DNS

Point your domain to the LoadBalancer IP and apply ingress:
```bash
kubectl apply -f deployment/k8s/ingress.yaml
```

## 💰 Cost Estimates

### Training Costs (GCP)

| Model | GPU | Time (10K samples, 3 epochs) | Cost |
|-------|-----|------------------------------|------|
| Llama 3.1 8B | A100-40GB | ~4 hours | ~$15 |
| Llama 3.1 70B | A100-80GB | ~20 hours | ~$105 |
| Mistral 7B | A100-40GB | ~3.5 hours | ~$13 |

### Inference Costs

| GPU | Cost/Hour | Throughput | Cost/1K tokens |
|-----|-----------|------------|----------------|
| L4 | $1.12 | ~500 req/hr | ~$0.002 |
| A100-40GB | $3.67 | ~1000 req/hr | ~$0.006 |

## 📊 API Reference

### Create Training Job

```bash
curl -X POST http://localhost:8000/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "job_name": "my-llama-finetune",
    "base_model": "meta-llama/Llama-3.1-8B",
    "dataset_path": "gs://my-bucket/data/train.jsonl",
    "hyperparameters": {
      "lora_r": 64,
      "lora_alpha": 16,
      "learning_rate": 0.0002,
      "num_epochs": 3
    },
    "gpu_type": "A100-40GB"
  }'
```

### Inference (OpenAI-compatible)

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is machine learning?",
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

## 🔬 Evaluation Metrics

LLMForge evaluates models with:

- **ROUGE-L**: Measures longest common subsequence
- **BLEU**: Measures n-gram overlap
- **Perplexity**: Measures model confidence
- **Exact Match**: Measures exact answer accuracy

Target metrics for quality fine-tuning:
- ROUGE-L > 0.5
- BLEU > 0.35
- Perplexity improvement > 20%

## 🛠️ Development

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Training (local)

```bash
cd training
pip install -r requirements.txt
python train.py sample_data.jsonl ./output
```

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) - Transformers, PEFT, TRL
- [vLLM](https://vllm.ai/) - High-throughput inference
- [QLoRA](https://arxiv.org/abs/2305.14314) - Efficient fine-tuning
- [Unsloth](https://unsloth.ai/) - Training optimizations
