#!/usr/bin/env python3
"""
Test Single Image Influence Computation
=======================================
This script tests the post-training influence computation for a specific image (22951)
using the same logic as compute_influence.py but focused on a single image.

Usage:
    python test_single_image_influence.py \
        --model_name_or_path ./path/to/model \
        --data_dir ./path/to/dataset \
        --target_image_id 22951 \
        --output_file single_image_test.jsonl
"""

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
    data_dir: str = field(
        metadata={"help": "The data directory containing input files."}
    )

@dataclass
class TestArguments:
    target_image_id: str = field(
        default="22951",
        metadata={"help": "Specific image ID to test"}
    )
    output_file: str = field(
        default="single_image_test.jsonl",
        metadata={"help": "File to save the influence result to."}
    )
    batch_size: int = field(
        default=1, 
        metadata={"help": "Batch size for processing. Should be 1 for true per-example grads."}
    )

def flatten_gradient_tree_by_layer(grads):
    """
    Extract gradients for key layers and compute per-neuron influence.
    
    Returns:
        dict: {layer_name: influence_vector} where influence_vector is a list of floats,
              one per neuron in that layer's output dimension.
    """
    def _flatten_grads(grads, prefix=''):
        if isinstance(grads, (dict, flax.core.FrozenDict)):
            items = grads.items() if isinstance(grads, dict) else grads.unfreeze().items()
            for k, v in items:
                yield from _flatten_grads(v, f'{prefix}{k}.')
        elif isinstance(grads, (list, tuple)):
            for i, v in enumerate(grads):
                yield from _flatten_grads(v, f'{prefix}{i}.')
        else:
            yield (prefix[:-1], grads)  # Remove trailing dot

    layer_influences = {}
    
    for param_path, grad_tensor in _flatten_grads(grads):
        if is_key_layer(param_path):
            layer_name = param_path  # Keep full path for EB compatibility
            influence_vector = compute_per_neuron_influence(grad_tensor, param_path)
            layer_influences[layer_name] = influence_vector
    
    return layer_influences

def is_key_layer(layer_path):
    """Check if this layer path corresponds to one of our key layers."""
    # Based on the training data, we need to match these exact patterns:
    key_patterns = [
        'GLU_0.Dense_0.kernel',  # fc1 (entrada) - appears as FlaxBartDecoderLayers.GLU_0.Dense_0.kernel
        'GLU_0.Dense_2.kernel',  # fc2 (salida) - appears as FlaxBartDecoderLayers.GLU_0.Dense_2.kernel  
        'lm_head.kernel'         # output head - appears as lm_head.kernel
    ]
    
    # Check if any pattern matches
    matches = any(pattern in layer_path for pattern in key_patterns)
    if matches:
        print(f"✅ Key layer detected: {layer_path}")
    
    return matches

def compute_per_neuron_influence(grad_tensor, layer_path):
    """
    Compute per-neuron influence for a gradient tensor.
    
    For FC layers, we compute the L2 norm per output neuron.
    This function handles various tensor shapes to ensure consistent per-neuron output.
    
    Args:
        grad_tensor: JAX array with gradients
        layer_path: String identifying the layer
    
    Returns:
        list: Per-neuron influence values (one per output neuron)
    """
    print(f"🔍 Processing {layer_path}: shape {grad_tensor.shape}, dtype {grad_tensor.dtype}")
    
    # Check for NaN or infinite values in gradients
    if jnp.any(jnp.isnan(grad_tensor)):
        print(f"⚠️ Warning: NaN values detected in gradients for {layer_path}")
        # Replace NaN with zeros
        grad_tensor = jnp.nan_to_num(grad_tensor, nan=0.0, posinf=0.0, neginf=0.0)
    
    if jnp.any(jnp.isinf(grad_tensor)):
        print(f"⚠️ Warning: Infinite values detected in gradients for {layer_path}")
        grad_tensor = jnp.nan_to_num(grad_tensor, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Handle different tensor shapes to ensure per-neuron computation
    if grad_tensor.ndim == 2:
        # Standard 2D weight matrix: (input_dim, output_dim)
        # Compute L2 norm along input dimension (axis=0) for each output neuron
        per_neuron_norms = jnp.linalg.norm(grad_tensor, axis=0)
        print(f"✅ 2D tensor: computed {len(per_neuron_norms)} neuron influences")
        
    elif grad_tensor.ndim == 3:
        # 3D tensor (common in BART): (num_layers, input_dim, output_dim)
        # We need to handle this properly to get per-neuron influences
        print(f"🔍 3D tensor detected: shape {grad_tensor.shape}")
        
        # For 3D tensors, we have multiple options:
        # Option 1: Flatten first two dimensions, then compute per output neuron
        # Option 2: Compute per layer, then aggregate
        # Option 3: Compute norm across all dimensions except the last (output neurons)
        
        # Using Option 3: compute norm across all dimensions except the last
        # This gives us one value per output neuron across all layers/inputs
        axes_to_norm = tuple(range(grad_tensor.ndim - 1))  # All axes except the last
        per_neuron_norms = jnp.linalg.norm(grad_tensor, axis=axes_to_norm)
        print(f"✅ 3D tensor: computed {len(per_neuron_norms)} neuron influences (norm across axes {axes_to_norm})")
        
    elif grad_tensor.ndim == 1:
        # 1D tensor: already per-neuron (bias terms or similar)
        per_neuron_norms = jnp.abs(grad_tensor)  # Use absolute value as "influence"
        print(f"✅ 1D tensor: using absolute values as influences ({len(per_neuron_norms)} neurons)")
        
    else:
        # For higher-dimensional tensors, compute norm across all but the last dimension
        print(f"🔍 {grad_tensor.ndim}D tensor detected: shape {grad_tensor.shape}")
        axes_to_norm = tuple(range(grad_tensor.ndim - 1))  # All axes except the last
        per_neuron_norms = jnp.linalg.norm(grad_tensor, axis=axes_to_norm)
        print(f"✅ {grad_tensor.ndim}D tensor: computed {len(per_neuron_norms)} neuron influences (norm across axes {axes_to_norm})")
    
    # Check for NaN in the result
    if jnp.any(jnp.isnan(per_neuron_norms)):
        print(f"⚠️ Warning: NaN in computed norms for {layer_path}, replacing with zeros")
        per_neuron_norms = jnp.nan_to_num(per_neuron_norms, nan=0.0)
    
    # Convert to Python list for JSON serialization
    result = per_neuron_norms.tolist()
    print(f"✅ Computed {len(result)} neuron influences for {layer_path}")
    print(f"   📊 Range: [{min(result):.6f}, {max(result):.6f}], Mean: {np.mean(result):.6f}")
    
    return result

def save_gradient_result(output_file, image_id, grads):
    """Save gradient result for one image to JSONL file."""
    layer_influences = flatten_gradient_tree_by_layer(grads)
    
    result = {
        "image_id": image_id,
        "layer_influences": layer_influences
    }
    
    # Append to JSONL file
    with open(output_file, 'a') as f:
        f.write(json.dumps(result) + '\n')
    
    print(f"✅ Saved influence data for image {image_id}")
    print(f"📊 Layers analyzed: {list(layer_influences.keys())}")
    for layer_name, influences in layer_influences.items():
        print(f"   🧠 {layer_name}: {len(influences)} neurons, max influence: {max(influences):.6f}")

def find_target_image_in_dataset(dataset, target_image_id):
    """Find the specific target image in the dataset."""
    print(f"🔍 Searching for image {target_image_id} in dataset...")
    
    for i, example in enumerate(dataset):
        if example.get('filename') == target_image_id:
            print(f"✅ Found target image {target_image_id} at index {i}")
            return example, i
    
    print(f"❌ Target image {target_image_id} not found in dataset")
    return None, -1

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TestArguments))
    model_args, data_args, test_args = parser.parse_args_into_dataclasses()

    print(f"🎯 Testing influence computation for image: {test_args.target_image_id}")
    print(f"📁 Model path: {model_args.model_name_or_path}")
    print(f"📁 Data path: {data_args.data_dir}")

    # 1. Load model and tokenizer
    print("🔄 Loading model and tokenizer...")
    
    model_path = os.path.abspath(model_args.model_name_or_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    tokenizer = DalleBartTokenizer.from_pretrained(model_path, local_files_only=True)
    model = DalleBart.from_pretrained(model_path, local_files_only=True)

    # 2. Initialize training state
    dummy_tx = optax.sgd(learning_rate=0.0)
    state = train_state.TrainState.create(apply_fn=model.__call__, params=model.params, tx=dummy_tx)

    # 3. Load dataset
    print("🔄 Loading dataset...")
    
    data_path = os.path.abspath(data_args.data_dir)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    
    parquet_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.parquet')]
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in {data_path}")
    
    print(f"📊 Found {len(parquet_files)} parquet files")
    
    dataset = load_dataset(
        'parquet',
        data_files=parquet_files,
        split='train',
        streaming=False
    )
    
    print(f"📊 Dataset loaded: {len(dataset)} examples")
    print(f"📊 Available columns: {dataset.column_names}")

    # 4. Find target image
    target_example, target_index = find_target_image_in_dataset(dataset, test_args.target_image_id)
    
    if target_example is None:
        print("❌ Target image not found. Exiting.")
        return

    # 5. Preprocess the target example
    print("🔄 Preprocessing target example...")
    
    # Check what data we have
    print(f"📊 Target example keys: {list(target_example.keys())}")
    
    if 'encoding' not in target_example:
        raise ValueError("'encoding' column not found in target example")
    
    # SOLUTION: Truncate sequence to avoid NaN issues with long sequences
    MAX_SAFE_LENGTH = 50  # Model works reliably up to length 50
    original_encoding = target_example['encoding']
    
    if len(original_encoding) > MAX_SAFE_LENGTH:
        print(f"⚠️ Truncating sequence from {len(original_encoding)} to {MAX_SAFE_LENGTH} tokens to avoid NaN issues")
        truncated_encoding = original_encoding[:MAX_SAFE_LENGTH]
    else:
        truncated_encoding = original_encoding
    
    # Prepare model input
    input_ids = jnp.array([truncated_encoding])  # Add batch dimension
    labels = jnp.array([truncated_encoding])     # Same for autoregressive
    
    print(f"📊 Original length: {len(original_encoding)}")
    print(f"📊 Truncated length: {len(truncated_encoding)}")
    print(f"📊 Input shape: {input_ids.shape}")
    print(f"📊 Labels shape: {labels.shape}")

    # 6. Define loss function and gradient computation
    def onehot(labels, num_classes):
        """Convert labels to one-hot encoding (same as training)."""
        return jnp.eye(num_classes)[labels]
    
    def loss_fn(logits, labels):
        """Loss function identical to training (using optax.softmax_cross_entropy)."""
        loss = optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1]))
        loss = loss.mean()
        return loss

    def compute_loss_single(params, batch):
        # Simple computation function without debugging (JAX grad-compatible)
        labels = batch['labels']
        model_input = {'input_ids': batch['input_ids']}
        logits = model(**model_input, params=params, train=False)[0]
        return loss_fn(logits, labels)

    grad_fn = jax.grad(compute_loss_single, argnums=0)

    # 7. Setup device
    print("🔄 Setting up computation device...")
    gpu_device = None
    try:
        gpu_device = jax.devices("gpu")[0]
        print(f"✅ Using GPU: {gpu_device}")
    except:
        print("⚠️ No GPU available, using CPU")
        gpu_device = jax.devices("cpu")[0]

    # Move model to device
    if gpu_device and "gpu" in str(gpu_device):
        print("🔄 Moving model parameters to GPU...")
        state = jax.device_put(state, gpu_device)

    # 8. Compute gradients for target image
    print(f"🔄 Computing gradients for image {test_args.target_image_id}...")
    
    model_input = {
        'input_ids': input_ids,
        'labels': labels
    }
    
    print(f"📊 Input shapes - input_ids: {input_ids.shape}, labels: {labels.shape}")
    print(f"📊 Input ranges - input_ids: [{jnp.min(input_ids)}, {jnp.max(input_ids)}]")
    
    # Move input to device
    if gpu_device:
        model_input = jax.tree_util.tree_map(
            lambda x: jax.device_put(x, gpu_device), 
            model_input
        )
    
    # Test model forward pass first (for debugging)
    print("🔄 Testing model forward pass...")
    try:
        test_logits = model(**{'input_ids': model_input['input_ids']}, params=state.params, train=False)[0]
        print(f"✅ Model forward pass successful - Logits shape: {test_logits.shape}")
        print(f"📊 Logits range: [{float(jnp.min(test_logits)):.6f}, {float(jnp.max(test_logits)):.6f}]")
        
        # Test loss computation
        test_loss = loss_fn(test_logits, model_input['labels'])
        print(f"✅ Loss computation successful - Loss: {float(test_loss):.6f}")
        
        if jnp.isnan(test_loss):
            print("⚠️ WARNING: Loss is NaN! This will cause gradient issues.")
        elif jnp.isinf(test_loss):
            print("⚠️ WARNING: Loss is infinite! This will cause gradient issues.")
        else:
            print("✅ Loss is finite and valid for gradient computation")
            
    except Exception as e:
        print(f"❌ Error in forward pass test: {e}")
        print("This error will likely cause gradient computation to fail.")
        raise
    
    # Compute gradients
    print("🔄 Computing gradients with jax.grad...")
    try:
        grads = grad_fn(state.params, model_input)
        print("✅ Gradient computation successful!")
    except Exception as e:
        print(f"❌ Error in gradient computation: {e}")
        raise
    
    # Move gradients back to CPU for storage
    grads_cpu = jax.tree_util.tree_map(lambda x: jax.device_put(x, jax.devices("cpu")[0]), grads)
    
    # 9. Save results
    print(f"🔄 Saving results to {test_args.output_file}...")
    
    # Clear output file first
    if os.path.exists(test_args.output_file):
        os.remove(test_args.output_file)
    
    save_gradient_result(test_args.output_file, test_args.target_image_id, grads_cpu)
    
    print("✅ Test completed successfully!")
    print(f"📄 Results saved to: {test_args.output_file}")
    
    # 10. Display summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    # Read and display the result
    with open(test_args.output_file, 'r') as f:
        result = json.loads(f.read().strip())
    
    print(f"🎯 Image ID: {result['image_id']}")
    print(f"🧠 Layers analyzed: {len(result['layer_influences'])}")
    
    for layer_name, influences in result['layer_influences'].items():
        influences_array = np.array(influences)
        print(f"\n📊 Layer: {layer_name}")
        print(f"   🔢 Neurons: {len(influences)}")
        print(f"   📈 Mean influence: {np.mean(influences_array):.6f}")
        print(f"   🔝 Max influence: {np.max(influences_array):.6f}")
        print(f"   🔻 Min influence: {np.min(influences_array):.6f}")
        print(f"   📊 Std deviation: {np.std(influences_array):.6f}")
        
        # Show top 5 neurons
        top_indices = np.argsort(influences_array)[-5:][::-1]
        print(f"   🏆 Top 5 neurons:")
        for i, idx in enumerate(top_indices, 1):
            print(f"      {i}. Neuron {idx}: {influences_array[idx]:.6f}")

if __name__ == "__main__":
    main()
