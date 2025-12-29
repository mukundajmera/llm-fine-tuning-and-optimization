"""
LLMForge Dataset Preparation
Dataset loading and formatting for instruction fine-tuning.

Supports:
- Local JSONL files
- Google Cloud Storage (GCS) paths
- Hugging Face Datasets

Expected input format (JSONL):
{
    "instruction": "What is machine learning?",
    "input": "",  // Optional context
    "output": "Machine learning is..."
}
"""

import os
import logging
from typing import Optional

from datasets import Dataset, load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Chat Templates for Different Models
# =============================================================================

LLAMA_3_1_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|}

{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{assistant_content}<|eot_id|>"""

MISTRAL_TEMPLATE = """<s>[INST] {user_content} [/INST] {assistant_content}</s>"""


def format_instruction_dataset(examples: dict, model_type: str = "llama") -> dict:
    """
    Convert dataset to model-specific instruction format.
    
    Args:
        examples: Batch of examples with 'instruction', 'input', 'output' keys
        model_type: One of 'llama' or 'mistral'
    
    Returns:
        dict with 'text' key containing formatted prompts
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
        
        # Apply model-specific template
        if model_type == "llama":
            formatted_text = LLAMA_3_1_TEMPLATE.format(
                user_content=user_content,
                assistant_content=output
            )
        elif model_type == "mistral":
            formatted_text = MISTRAL_TEMPLATE.format(
                user_content=user_content,
                assistant_content=output
            )
        else:
            # Generic format
            formatted_text = f"### Instruction:\n{user_content}\n\n### Response:\n{output}"
        
        formatted_texts.append(formatted_text)
    
    return {"text": formatted_texts}


def download_from_gcs(gcs_path: str) -> str:
    """
    Download file from Google Cloud Storage.
    
    Args:
        gcs_path: GCS path (gs://bucket/path/to/file)
    
    Returns:
        Local path to downloaded file
    """
    from google.cloud import storage
    
    # Parse bucket and blob path
    path_without_prefix = gcs_path.replace("gs://", "")
    bucket_name, blob_path = path_without_prefix.split("/", 1)
    
    # Download file
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    
    # Save to temp location
    local_path = f"/tmp/{os.path.basename(blob_path)}"
    blob.download_to_filename(local_path)
    
    logger.info(f"Downloaded {gcs_path} to {local_path}")
    return local_path


def load_and_prepare_dataset(
    dataset_path: str,
    model_type: str = "llama",
    split_ratio: float = 0.1,
    max_samples: Optional[int] = None,
    seed: int = 42
) -> tuple[Dataset, Dataset]:
    """
    Load dataset from GCS/local, format, and split.
    
    Args:
        dataset_path: Path to JSONL file (local or gs://)
        model_type: One of 'llama' or 'mistral'
        split_ratio: Fraction of data for evaluation
        max_samples: Maximum number of samples to use (for debugging)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    logger.info(f"Loading dataset from: {dataset_path}")
    
    # Handle GCS paths
    if dataset_path.startswith("gs://"):
        local_path = download_from_gcs(dataset_path)
        dataset = load_dataset('json', data_files=local_path, split='train')
    else:
        dataset = load_dataset('json', data_files=dataset_path, split='train')
    
    logger.info(f"Loaded {len(dataset)} samples")
    
    # Validate required columns
    required_columns = {'instruction', 'output'}
    if not required_columns.issubset(set(dataset.column_names)):
        raise ValueError(
            f"Dataset must have columns: {required_columns}. "
            f"Found: {dataset.column_names}"
        )
    
    # Limit samples if specified
    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
        logger.info(f"Limited to {max_samples} samples")
    
    # Format dataset
    dataset = dataset.map(
        lambda x: format_instruction_dataset(x, model_type=model_type),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Formatting dataset"
    )
    
    # Split train/eval
    split_dataset = dataset.train_test_split(
        test_size=split_ratio,
        seed=seed
    )
    
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    logger.info(
        f"✅ Dataset prepared: {len(train_dataset)} train, "
        f"{len(eval_dataset)} eval samples"
    )
    
    return train_dataset, eval_dataset


def create_sample_dataset(output_path: str, num_samples: int = 100) -> str:
    """
    Create a sample instruction dataset for testing.
    
    Args:
        output_path: Path to save the JSONL file
        num_samples: Number of samples to generate
    
    Returns:
        Path to the created file
    """
    import json
    
    samples = []
    for i in range(num_samples):
        samples.append({
            "instruction": f"Explain concept number {i} in machine learning.",
            "input": "",
            "output": f"Concept {i} is an important topic in machine learning that involves understanding patterns in data. It helps models learn from examples and make predictions on new data."
        })
    
    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    logger.info(f"Created sample dataset with {num_samples} samples at {output_path}")
    return output_path


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python data_prep.py <dataset_path> [model_type]")
        print("       python data_prep.py --create-sample <output_path>")
        sys.exit(1)
    
    if sys.argv[1] == "--create-sample":
        output_path = sys.argv[2] if len(sys.argv) > 2 else "sample_dataset.jsonl"
        create_sample_dataset(output_path)
    else:
        dataset_path = sys.argv[1]
        model_type = sys.argv[2] if len(sys.argv) > 2 else "llama"
        
        train_dataset, eval_dataset = load_and_prepare_dataset(
            dataset_path,
            model_type=model_type
        )
        
        # Print sample
        print("\nSample formatted text:")
        print(train_dataset[0]['text'][:500])
