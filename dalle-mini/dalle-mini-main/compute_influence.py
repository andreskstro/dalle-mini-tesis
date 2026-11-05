import os
import gc
import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

import jax
import jax.numpy as jnp
import optax
import numpy as np
import torch
import flax
import flax.core
from flax.training import train_state
from transformers import HfArgumentParser, AutoTokenizer, FlaxAutoModelForCausalLM
from datasets import load_dataset

from dalle_mini.model import (  
    DalleBart,
    DalleBartConfig,
    DalleBartTokenizer
)

# Define arguments
@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    data_dir: Optional[str] = field(default=None, metadata={"help": "The data directory containing input files."})

@dataclass
class InfluenceArguments:
    output_file: str = field(
        default="influence_map.jsonl",
        metadata={"help": "File to save the influence map to."}
    )
    batch_size: int = field(
        default=1, 
        metadata={"help": "Batch size for processing. Should be 1 for true per-example grads."}
    )
    max_seq_length: int = field(
        default=1024,
        metadata={"help": "Maximum sequence length for preprocessing."}
    )
    is_encoder_decoder: bool = field(
        default=True,
        metadata={"help": "Whether the model is encoder-decoder."}
    )
    input_columns: str = field(
        default="input_ids",
        metadata={"help": "Input column name."}
    )
    target_columns: str = field(
        default="input_ids",
        metadata={"help": "Target column name."}
    )
    num_workers: int = field(
        default=4,
        metadata={"help": "Number of workers for data loading."}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed."}
    )


def flatten_gradient_tree_by_layer(grads):
    """
    Extract gradients for key layers and compute per-neuron influence.
    
    Returns:
        dict: {layer_name: influence_vector} where influence_vector is a list of floats,
              one per neuron in that layer's output dimension.
    """
    layer_influences = {}
    all_param_names = []  # Debug: collect all parameter names
    
    def extract_layer_gradients(grad_tree, prefix=''):
        if isinstance(grad_tree, (dict, flax.core.FrozenDict)):
            items = grad_tree.items() if isinstance(grad_tree, dict) else grad_tree.unfreeze().items()
            for k, v in items:
                new_prefix = f'{prefix}.{k}' if prefix else k
                extract_layer_gradients(v, new_prefix)
        else:
            # This is a leaf node (actual gradient tensor)
            all_param_names.append(prefix)  # Debug: collect parameter name
            if is_key_layer(prefix):
                # Compute per-neuron influence for this layer
                influence_vector = compute_per_neuron_influence(grad_tree, prefix)
                layer_influences[prefix] = influence_vector
                print(f"DEBUG: Found key layer '{prefix}' with shape {grad_tree.shape}")
    
    extract_layer_gradients(grads)
    
    # Debug: Print all parameter names to understand the structure
    print("DEBUG: All parameter names in gradient tree:")
    for name in sorted(all_param_names):
        print(f"  {name}")
    
    print(f"DEBUG: Found {len(layer_influences)} key layers out of {len(all_param_names)} total parameters")
    
    return layer_influences

def is_key_layer(layer_path):
    """Check if this layer path corresponds to one of our key layers."""
    
    # Check for decoder GLU Dense layers (equivalent to fc1, fc2 layers)
    if ('model.decoder.layers.' in layer_path and 
        'GLU_0.Dense_' in layer_path and 
        '.kernel' in layer_path):
        return True
    
    # Check for encoder GLU Dense layers (also important for influence)
    if ('model.encoder.layers.' in layer_path and 
        'GLU_0.Dense_' in layer_path and 
        '.kernel' in layer_path):
        return True
    
    # Check for lm_head (output layer)
    if layer_path == 'lm_head.kernel':
        return True
    
    # Check for attention projection layers (also important)
    if (('model.decoder.layers.' in layer_path or 'model.encoder.layers.' in layer_path) and
        ('q_proj.kernel' in layer_path or 'k_proj.kernel' in layer_path or 
         'v_proj.kernel' in layer_path or 'out_proj.kernel' in layer_path)):
        return True
        
    return False

def compute_per_neuron_influence(grad_tensor, layer_path):
    """
    Compute per-neuron influence for a gradient tensor.
    
    For FC layers (shape: [input_dim, output_dim]), we compute the L2 norm
    across the input dimension for each output neuron.
    
    Args:
        grad_tensor: JAX array with gradients
        layer_path: String identifying the layer
    
    Returns:
        list: Per-neuron influence values (one per output neuron)
    """
    # Move to CPU for computation
    grad_cpu = jax.device_get(grad_tensor)
    
    # Check for problematic values first
    if jnp.isnan(grad_cpu).any() or jnp.isinf(grad_cpu).any():
        print(f"WARNING: Found NaN/Inf in gradients for layer '{layer_path}', replacing with zeros")
        grad_cpu = jnp.nan_to_num(grad_cpu, nan=0.0, posinf=0.0, neginf=0.0)
    
    if len(grad_cpu.shape) == 2:  # Standard FC layer: [input_dim, output_dim]
        # Compute L2 norm across input dimension (axis=0) for each output neuron
        per_neuron_influence = jnp.linalg.norm(grad_cpu, axis=0)
    elif len(grad_cpu.shape) == 1:  # Bias vector: [output_dim]
        # For bias, each element directly represents one neuron
        per_neuron_influence = jnp.abs(grad_cpu)
    else:
        # For other shapes, flatten and take overall norm (fallback)
        per_neuron_influence = jnp.array([jnp.linalg.norm(grad_cpu.flatten())])
    
    # Final check for NaN values after computation
    if jnp.isnan(per_neuron_influence).any():
        print(f"WARNING: NaN values found in per-neuron influence for layer '{layer_path}' after computation")
        per_neuron_influence = jnp.nan_to_num(per_neuron_influence, nan=0.0)
    
    # Check if all values are zero (might indicate a problem)
    if jnp.allclose(per_neuron_influence, 0.0):
        print(f"INFO: All influence values are zero for layer '{layer_path}' - this might be normal for some layers")
    
    # Convert to Python list for JSON serialization
    return per_neuron_influence.tolist()


def get_processed_image_ids(output_file):
    """
    Read the existing JSONL file and return a set of already processed image_ids.
    
    Args:
        output_file: Path to the JSONL output file
    
    Returns:
        Set of processed image_ids
    """
    processed_ids = set()
    
    if os.path.exists(output_file):
        print(f"Reading existing results from {output_file}...")
        with open(output_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    if 'image_id' in data:
                        processed_ids.add(data['image_id'])
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed line {line_num}: {e}")
        
        print(f"Found {len(processed_ids)} already processed images.")
    
    return processed_ids


def save_gradient_result(output_file, image_id, grads):
    """Save gradient result for one image to JSONL file."""
    # Extract layer-wise influences instead of flattening all parameters
    layer_influences = flatten_gradient_tree_by_layer(grads)
    
    result = {
        'image_id': image_id,
        'layer_influences': layer_influences
    }
    
    # Append to JSONL file
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result) + '\n')
    
    # Log progress with layer info
    num_layers = len(layer_influences)
    total_neurons = sum(len(influence_vec) for influence_vec in layer_influences.values())
    print(f"Saved influences for image {image_id}: {num_layers} layers, {total_neurons} total neurons")


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, InfluenceArguments))
    model_args, data_args, influence_args = parser.parse_args_into_dataclasses()

    # 1. Load model and tokenizer
    print("Loading model and tokenizer...")
    
    # Convert relative path to absolute path for local loading
    model_path = os.path.abspath(model_args.model_name_or_path)
    print(f"Model path: {model_path}")
    
    # Check if the path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    # Load tokenizer and model from local path
    tokenizer = DalleBartTokenizer.from_pretrained(model_path, local_files_only=True)
    model = DalleBart.from_pretrained(model_path, local_files_only=True)

    # 2. Initialize training state
    dummy_tx = optax.sgd(learning_rate=0.0)
    state = train_state.TrainState.create(apply_fn=model.__call__, params=model.params, tx=dummy_tx)

    # 3. Load dataset
    print("Loading dataset...")
    
    # Convert relative data path to absolute path
    if not data_args.data_dir:
        raise ValueError("data_dir must be specified")
        
    data_path = os.path.abspath(data_args.data_dir)
    print(f"Data path: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    
    # Get full paths to parquet files
    parquet_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.parquet')]
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in {data_path}")
    
    print(f"Found {len(parquet_files)} parquet files in dataset directory")
    
    # Load dataset directly with load_dataset
    print(f"Loading dataset from {len(parquet_files)} parquet files...")
    dataset = load_dataset(
        'parquet',
        data_files=parquet_files,
        split='train',
        streaming=False
    )
    
    # Debug: Print available columns
    print(f"Available columns in dataset: {dataset.column_names}")
    print(f"Dataset info: {dataset}")
    
    # Show a sample of the first few examples to understand the structure
    if len(dataset) > 0:
        print("Sample data:")
        sample = dataset[0]
        for key, value in sample.items():
            if isinstance(value, (list, tuple)) and len(value) > 10:
                print(f"  {key}: {type(value)} with {len(value)} elements (showing first 10): {value[:10]}...")
            else:
                print(f"  {key}: {value}")
    
    # Preprocess the dataset
    def preprocess_function(examples):
        # The parquet files contain 'encoding' as the tokenized data
        processed = {}
        
        # Check what columns are actually available
        available_keys = list(examples.keys())
        print(f"Available keys in batch: {available_keys}")
        
        # Use 'encoding' column as input_ids (this contains the tokenized data)
        if 'encoding' in examples:
            processed['input_ids'] = examples['encoding']
            processed['labels'] = examples['encoding']  # For autoregressive models
        else:
            raise ValueError(f"'encoding' column not found. Available columns: {available_keys}")
            
        # Use 'filename' for influence tracking
        if 'filename' in examples:
            processed['filename'] = examples['filename']
        else:
            # Generate dummy filenames if not present
            processed['filename'] = [f"example_{i}" for i in range(len(examples['encoding']))]
            
        return processed

    dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=[col for col in dataset.column_names if col not in ['input_ids', 'labels', 'filename']]
    )
    
    # Create DataLoader
    def collate_fn(batch):
        # Convert to JAX arrays
        input_ids = jnp.array([item['input_ids'] for item in batch])
        labels = jnp.array([item['labels'] for item in batch])
        filenames = [item['filename'] for item in batch]
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'filenames': filenames
        }
    
    # Create batched dataset
    dataset = dataset.with_format("numpy")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=influence_args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Get processed image IDs
    processed_ids = get_processed_image_ids(influence_args.output_file)

    # Define loss function (cross-entropy for token prediction)
    def loss_fn(logits, labels):
        # logits: (batch_size, seq_len, vocab_size)
        # labels: (batch_size, seq_len) - token indices
        
        # Reshape for cross-entropy calculation
        logits_flat = logits.reshape(-1, logits.shape[-1])  # (batch_size * seq_len, vocab_size)
        labels_flat = labels.reshape(-1)  # (batch_size * seq_len,)
        
        # Compute cross-entropy loss
        log_probs = jax.nn.log_softmax(logits_flat, axis=-1)
        loss = -jnp.mean(log_probs[jnp.arange(labels_flat.shape[0]), labels_flat])
        
        return loss

    # Define function to compute loss for a single example
    def compute_loss_single(params, batch):
        labels = batch['labels']
        
        # Only pass input_ids to the model (not labels)
        model_input = {'input_ids': batch['input_ids']}
        logits = model(**model_input, params=params, train=False)[0]
        
        return loss_fn(logits, labels)

    grad_fn = jax.grad(compute_loss_single, argnums=0)

    # Check available devices
    print(f"Available devices: {jax.devices()}")
    gpu_device = None
    try:
        gpu_device = jax.devices("gpu")[0]
        print(f"Using GPU: {gpu_device}")
    except:
        print("No GPU available, using CPU")
        gpu_device = jax.devices("cpu")[0]

    # Move model parameters to GPU if available
    if gpu_device and "gpu" in str(gpu_device):
        print("Moving model parameters to GPU...")
        state = jax.device_put(state, gpu_device)
        print(f"Params device: {jax.tree_util.tree_map(lambda x: x.device(), state.params)}")

    # 5. Iterate over the dataset
    print("Starting influence computation...")
    for i, batch in enumerate(dataloader):
        # Ensure filenames are present
        if 'filenames' not in batch:
            print(f"Skipping batch {i} because 'filenames' is missing.")
            continue

        # Get the filename for this batch (assuming batch size 1)
        image_id = batch['filenames'][0]
        
        # Skip if already processed
        if image_id in processed_ids:
            print(f"Skipping already processed image: {image_id}")
            continue
        
        print(f"Processing {image_id}...")
        
        # Prepare input for the model and move to GPU
        model_input = {
            'input_ids': batch['input_ids'],
            'labels': batch['labels']
        }
        
        # Move input to GPU if available
        if gpu_device:
            model_input = jax.tree_util.tree_map(
                lambda x: jax.device_put(x, gpu_device), 
                model_input
            )
        
        # Debug: Print device info for first batch
        if i == 0:
            print(f"Input device: {jax.tree_util.tree_map(lambda x: x.device(), model_input)}")
        
        # Compute gradients for this example
        grads = grad_fn(state.params, model_input)
        
        # Move gradients back to CPU for storage (to save GPU memory)
        grads_cpu = jax.tree_util.tree_map(lambda x: jax.device_put(x, jax.devices("cpu")[0]), grads)
        
        # Save gradient result
        save_gradient_result(influence_args.output_file, image_id, grads_cpu)
        
        # Periodic garbage collection and progress reporting
        if i % 100 == 0:
            gc.collect()
            print(f"Processed {i+1} examples...")
        elif i % 10 == 0:  # More frequent progress for debugging
            print(f"Processed {i+1} examples...")
    
    print("Influence computation completed!")

if __name__ == "__main__":
    main()
