#!/usr/bin/env python3
"""
Test Training Context Influence Computation
===========================================
This script replicates the exact training context to compute neuron influences
for image 22951 using full 256-token sequences, just like during training.
"""

import os
import gc
import json
import jax
import jax.numpy as jnp
import optax
import numpy as np
from datasets import load_dataset
from dalle_mini.model import DalleBart, DalleBartTokenizer
from flax.training import train_state
from flax.core import freeze

def load_image_22951():
    """Load the specific image 22951 from the dataset."""
    data_path = "tools/dataset/encoded_data/train"
    
    # Get parquet files
    parquet_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.parquet')]
    
    # Load dataset
    dataset = load_dataset('parquet', data_files=parquet_files, split='train', streaming=False)
    
    # Find image 22951
    for example in dataset:
        if example.get('filename') == '22951':
            return example
    
    return None

def create_training_batch(example, batch_size=1):
    """Create a batch in the same format as training."""
    encoding = example['encoding']
    
    # Create batch with proper structure (same as training)
    batch = {
        'input_ids': jnp.array([encoding] * batch_size),  # Shape: (batch_size, seq_len)
        'labels': jnp.array([encoding] * batch_size)      # Shape: (batch_size, seq_len)
    }
    
    return batch

def replicate_training_gradient_computation(model, batch):
    """
    Replicate the exact gradient computation from training.
    Based on train.py lines 1187-1228.
    """
    
    # 1. Define loss function (same as training)
    def onehot(labels, num_classes):
        return jnp.eye(num_classes)[labels]
    
    def loss_fn(logits, labels):
        """Exact same loss function as training."""
        loss = optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1]))
        loss = loss.mean()
        return loss
    
    # 2. Define compute_loss function (same structure as training)
    def compute_loss(params, minibatch, dropout_rng=None):
        """
        Replicate the exact compute_loss from training (line 1187-1195).
        """
        # Reshape minibatch (same as training line 1188-1190)
        minibatch = jax.tree_util.tree_map(
            lambda x: x.reshape(-1, x.shape[-1]), minibatch
        )
        
        # Extract labels (same as training line 1191)
        labels = minibatch.pop("labels")
        
        # Apply model (same as training line 1192-1194)
        # Note: using train=True just like in training!
        logits = model(**minibatch, params=params, train=True)[0]
        
        return loss_fn(logits, labels)
    
    # 3. Create gradient function (same as training line 1197)
    grad_fn = jax.value_and_grad(compute_loss)
    
    # 4. Compute loss and gradients (same as training line 1224)
    # Note: We're not using dropout_rng for simplicity, but structure is the same
    loss, grads = grad_fn(model.params, batch, None)
    
    return loss, grads

def find_target_layer(tree, target_pattern="GLU_0.Dense_0.kernel", path=""):
    """Find target layer in gradient tree."""
    if isinstance(tree, dict):
        for key, value in tree.items():
            current_path = f"{path}.{key}" if path else key
            if target_pattern in current_path:
                return value, current_path
            result, result_path = find_target_layer(value, target_pattern, current_path)
            if result is not None:
                return result, result_path
    elif hasattr(tree, 'items'):  # Handle other dict-like objects
        for key, value in tree.items():
            current_path = f"{path}.{key}" if path else str(key)
            if target_pattern in current_path:
                return value, current_path
            result, result_path = find_target_layer(value, target_pattern, current_path)
            if result is not None:
                return result, result_path
    return None, None

def compute_per_neuron_influence(grad_tensor, layer_path):
    """
    Compute per-neuron influence using the same method as training.
    """
    print(f"🔍 Processing {layer_path}: shape {grad_tensor.shape}, dtype {grad_tensor.dtype}")
    
    # Handle NaN/Inf (same as our fixed version)
    if jnp.any(jnp.isnan(grad_tensor)):
        print(f"⚠️ Warning: NaN values detected in gradients for {layer_path}")
        grad_tensor = jnp.nan_to_num(grad_tensor, nan=0.0, posinf=0.0, neginf=0.0)
    
    if jnp.any(jnp.isinf(grad_tensor)):
        print(f"⚠️ Warning: Infinite values detected in gradients for {layer_path}")
        grad_tensor = jnp.nan_to_num(grad_tensor, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Handle different tensor shapes (same as our fixed version)
    if grad_tensor.ndim == 2:
        # Standard 2D weight matrix: (input_dim, output_dim)
        per_neuron_norms = jnp.linalg.norm(grad_tensor, axis=0)
        print(f"✅ 2D tensor: computed {len(per_neuron_norms)} neuron influences")
        
    elif grad_tensor.ndim == 3:
        # 3D tensor (common in BART): (num_layers, input_dim, output_dim)
        print(f"🔍 3D tensor detected: shape {grad_tensor.shape}")
        axes_to_norm = tuple(range(grad_tensor.ndim - 1))  # All axes except the last
        per_neuron_norms = jnp.linalg.norm(grad_tensor, axis=axes_to_norm)
        print(f"✅ 3D tensor: computed {len(per_neuron_norms)} neuron influences (norm across axes {axes_to_norm})")
        
    elif grad_tensor.ndim == 1:
        # 1D tensor: already per-neuron
        per_neuron_norms = jnp.abs(grad_tensor)
        print(f"✅ 1D tensor: using absolute values as influences ({len(per_neuron_norms)} neurons)")
        
    else:
        # Higher-dimensional tensors
        print(f"🔍 {grad_tensor.ndim}D tensor detected: shape {grad_tensor.shape}")
        axes_to_norm = tuple(range(grad_tensor.ndim - 1))
        per_neuron_norms = jnp.linalg.norm(grad_tensor, axis=axes_to_norm)
        print(f"✅ {grad_tensor.ndim}D tensor: computed {len(per_neuron_norms)} neuron influences")
    
    # Check for NaN in result
    if jnp.any(jnp.isnan(per_neuron_norms)):
        print(f"⚠️ Warning: NaN in computed norms for {layer_path}, replacing with zeros")
        per_neuron_norms = jnp.nan_to_num(per_neuron_norms, nan=0.0)
    
    # Convert to Python list
    result = per_neuron_norms.tolist()
    print(f"✅ Computed {len(result)} neuron influences for {layer_path}")
    print(f"   📊 Range: [{min(result):.6f}, {max(result):.6f}], Mean: {np.mean(result):.6f}")
    
    return result

def main():
    print("🔍 TESTING TRAINING CONTEXT INFLUENCE COMPUTATION")
    print("=" * 60)
    
    # Load model and tokenizer
    print("🔄 Loading model and tokenizer...")
    model_path = "outputs/dalle-mini-from-scratch"
    model = DalleBart.from_pretrained(model_path, local_files_only=True)
    tokenizer = DalleBartTokenizer.from_pretrained(model_path, local_files_only=True)
    
    # Load image 22951
    print("🔄 Loading image 22951...")
    image_data = load_image_22951()
    
    if not image_data:
        print("❌ Image 22951 not found")
        return
    
    original_encoding = image_data['encoding']
    print(f"✅ Loaded image 22951: {len(original_encoding)} tokens")
    print(f"📊 Caption: {image_data.get('caption', 'N/A')}")
    
    # Test different approaches
    approaches = [
        ("Full Sequence (256 tokens) - Training Context", len(original_encoding)),
        ("Truncated Sequence (50 tokens) - Training Context", 50),
    ]
    
    results = {}
    
    for approach_name, seq_length in approaches:
        print(f"\n🔬 Testing: {approach_name}")
        print("=" * 50)
        
        # Prepare sequence
        if seq_length < len(original_encoding):
            encoding = original_encoding[:seq_length]
            print(f"⚠️ Truncating sequence from {len(original_encoding)} to {seq_length} tokens")
        else:
            encoding = original_encoding
        
        # Create training-style batch
        batch = create_training_batch({'encoding': encoding})
        print(f"📊 Batch shape: input_ids={batch['input_ids'].shape}, labels={batch['labels'].shape}")
        
        try:
            # Compute gradients using training context
            print("🔄 Computing gradients with training context...")
            loss, grads = replicate_training_gradient_computation(model, batch)
            
            print(f"✅ Loss computed: {float(loss):.6f}")
            
            # Check if loss is valid
            if jnp.isnan(loss):
                print("❌ Loss is NaN - gradient computation failed")
                results[approach_name] = {"status": "failed", "reason": "NaN loss"}
                continue
            
            # Find target layer
            target_layer, layer_path = find_target_layer(grads, "GLU_0.Dense_0.kernel")
            
            if target_layer is not None:
                print(f"✅ Found target layer: {layer_path}")
                
                # Compute per-neuron influences
                influences = compute_per_neuron_influence(target_layer, layer_path)
                
                results[approach_name] = {
                    "status": "success",
                    "seq_length": seq_length,
                    "loss": float(loss),
                    "layer_path": layer_path,
                    "tensor_shape": target_layer.shape,
                    "num_neurons": len(influences),
                    "max_influence": max(influences),
                    "mean_influence": np.mean(influences),
                    "min_influence": min(influences),
                    "std_influence": np.std(influences)
                }
                
                print(f"🎯 Results Summary:")
                print(f"   Max: {max(influences):.6f}")
                print(f"   Mean: {np.mean(influences):.6f}")
                print(f"   Min: {min(influences):.6f}")
                print(f"   Std: {np.std(influences):.6f}")
                
            else:
                print("❌ Target layer not found in gradients")
                results[approach_name] = {"status": "failed", "reason": "Target layer not found"}
                
        except Exception as e:
            print(f"❌ Error during gradient computation: {e}")
            results[approach_name] = {"status": "failed", "reason": str(e)}
    
    # Final comparison
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    
    print("\n🎯 Results Summary:")
    for approach, result in results.items():
        print(f"\n📊 {approach}:")
        if result["status"] == "success":
            print(f"   ✅ Success - Seq Length: {result['seq_length']}")
            print(f"   📈 Max: {result['max_influence']:.6f}")
            print(f"   📊 Mean: {result['mean_influence']:.6f}")
            print(f"   🔢 Neurons: {result['num_neurons']}")
            print(f"   📉 Loss: {result['loss']:.6f}")
        else:
            print(f"   ❌ Failed: {result['reason']}")
    
    # Compare with training data
    print(f"\n🔄 Comparison with Training Data:")
    print(f"   Training (256 tokens): Max=0.316406, Mean=0.033972")
    
    for approach, result in results.items():
        if result["status"] == "success":
            ratio_max = result['max_influence'] / 0.316406
            ratio_mean = result['mean_influence'] / 0.033972
            print(f"   {approach}:")
            print(f"      Max ratio: {ratio_max:.2f}x")
            print(f"      Mean ratio: {ratio_mean:.2f}x")

if __name__ == "__main__":
    main()
