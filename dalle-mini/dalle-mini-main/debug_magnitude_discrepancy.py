#!/usr/bin/env python3
"""
Debug Magnitude Discrepancy Script
==================================
This script investigates why there's a 10x magnitude difference between
training and post-training neuron influences for the same image.
"""

import os
import json
import jax
import jax.numpy as jnp
import optax
import numpy as np
from datasets import load_dataset
from dalle_mini.model import DalleBart, DalleBartTokenizer

def load_training_influences():
    """Load the training influence data for image 22951."""
    influence_file = "eb_neuron_influences.json"
    
    if not os.path.exists(influence_file):
        print(f"❌ Training influence file not found: {influence_file}")
        return None
    
    try:
        with open(influence_file, 'r') as f:
            # First, try to load as a single JSON object/array
            content = f.read().strip()
            
            if content.startswith('['):
                # It's a JSON array
                data = json.loads(content)
                for record in data:
                    if record.get('image_id') == '22951':
                        print(f"✅ Found image 22951 in JSON array")
                        return record
                        
            elif content.startswith('{'):
                # It's a single JSON object
                data = json.loads(content)
                if data.get('image_id') == '22951':
                    print(f"✅ Found image 22951 as single JSON object")
                    return data
                    
            else:
                # Try as JSONL (JSON Lines) - each line is a separate JSON object
                print(f"🔄 Trying as JSONL format...")
                f.seek(0)  # Reset file pointer
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if record.get('image_id') == '22951':
                            print(f"✅ Found image 22951 in JSONL line {line_num}")
                            return record
                    except json.JSONDecodeError as e:
                        print(f"⚠️ JSON error in line {line_num}: {e}")
                        continue
        
        print("❌ Image 22951 not found in training influences")
        return None
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
        return None
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None

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

def test_different_sequence_lengths(model, tokenizer, original_encoding):
    """Test how gradient magnitudes change with different sequence lengths."""
    print("\n🔍 Testing Different Sequence Lengths:")
    print("=" * 50)
    
    def onehot(labels, num_classes):
        return jnp.eye(num_classes)[labels]
    
    def loss_fn(logits, labels):
        loss = optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1]))
        return loss.mean()
    
    def compute_loss_single(params, batch):
        labels = batch['labels']
        model_input = {'input_ids': batch['input_ids']}
        logits = model(**model_input, params=params, train=False)[0]
        return loss_fn(logits, labels)
    
    grad_fn = jax.grad(compute_loss_single, argnums=0)
    
    # Test different lengths
    lengths_to_test = [10, 25, 50, 100, 150, 200, 256]
    results = {}
    
    for length in lengths_to_test:
        if length > len(original_encoding):
            continue
            
        print(f"\n📏 Testing length {length}:")
        
        # Prepare input
        truncated_encoding = original_encoding[:length]
        input_ids = jnp.array([truncated_encoding])
        labels = jnp.array([truncated_encoding])
        
        batch = {
            'input_ids': input_ids,
            'labels': labels
        }
        
        try:
            # Compute gradients
            grads = grad_fn(model.params, batch)
            
            # Find the GLU_0.Dense_0.kernel layer using tree_flatten
            target_layer = None
            flat_grads, tree_def = jax.tree_util.tree_flatten(grads)
            flat_paths = jax.tree_util.tree_flatten(jax.tree_util.tree_map(lambda x: "placeholder", grads))[0]
            
            # Alternative approach: use the same method as in the working script
            def find_target_layer(tree, path=""):
                if isinstance(tree, dict):
                    for key, value in tree.items():
                        current_path = f"{path}.{key}" if path else key
                        if 'GLU_0' in current_path and 'Dense_0' in current_path and 'kernel' in current_path:
                            return value
                        result = find_target_layer(value, current_path)
                        if result is not None:
                            return result
                elif hasattr(tree, 'items'):  # Handle other dict-like objects
                    for key, value in tree.items():
                        current_path = f"{path}.{key}" if path else str(key)
                        if 'GLU_0' in current_path and 'Dense_0' in current_path and 'kernel' in current_path:
                            return value
                        result = find_target_layer(value, current_path)
                        if result is not None:
                            return result
                return None
            
            target_layer = find_target_layer(grads)
            
            if target_layer is not None:
                # Compute per-neuron influences
                if target_layer.ndim == 3:
                    # 3D tensor: compute norm across all but last dimension
                    axes_to_norm = tuple(range(target_layer.ndim - 1))
                    per_neuron_norms = jnp.linalg.norm(target_layer, axis=axes_to_norm)
                elif target_layer.ndim == 2:
                    # 2D tensor: compute norm along input dimension
                    per_neuron_norms = jnp.linalg.norm(target_layer, axis=0)
                else:
                    per_neuron_norms = jnp.abs(target_layer)
                
                # Statistics
                max_influence = float(jnp.max(per_neuron_norms))
                mean_influence = float(jnp.mean(per_neuron_norms))
                
                results[length] = {
                    'max': max_influence,
                    'mean': mean_influence,
                    'shape': target_layer.shape,
                    'neurons': len(per_neuron_norms)
                }
                
                print(f"   ✅ Max: {max_influence:.6f}, Mean: {mean_influence:.6f}")
                print(f"   📊 Shape: {target_layer.shape}, Neurons: {len(per_neuron_norms)}")
                
            else:
                print(f"   ❌ Target layer not found")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            if length >= 100:
                print(f"   💡 This confirms the NaN issue at length {length}")
    
    return results

def compare_train_vs_inference_mode(model, original_encoding):
    """Compare gradients computed in train vs inference mode."""
    print("\n🔍 Comparing Train vs Inference Mode:")
    print("=" * 50)
    
    def onehot(labels, num_classes):
        return jnp.eye(num_classes)[labels]
    
    def loss_fn(logits, labels):
        loss = optax.softmax_cross_entropy(logits, onehot(labels, logits.shape[-1]))
        return loss.mean()
    
    def compute_loss_train(params, batch):
        labels = batch['labels']
        model_input = {'input_ids': batch['input_ids']}
        logits = model(**model_input, params=params, train=True)[0]  # train=True
        return loss_fn(logits, labels)
    
    def compute_loss_inference(params, batch):
        labels = batch['labels']
        model_input = {'input_ids': batch['input_ids']}
        logits = model(**model_input, params=params, train=False)[0]  # train=False
        return loss_fn(logits, labels)
    
    grad_fn_train = jax.grad(compute_loss_train, argnums=0)
    grad_fn_inference = jax.grad(compute_loss_inference, argnums=0)
    
    # Use safe length (50 tokens)
    truncated_encoding = original_encoding[:50]
    input_ids = jnp.array([truncated_encoding])
    labels = jnp.array([truncated_encoding])
    
    batch = {
        'input_ids': input_ids,
        'labels': labels
    }
    
    # Compute gradients in both modes
    print("🔄 Computing gradients in train mode...")
    grads_train = grad_fn_train(model.params, batch)
    
    print("🔄 Computing gradients in inference mode...")
    grads_inference = grad_fn_inference(model.params, batch)
    
    # Compare the target layer using the same find function
    def find_target_layer(tree, path=""):
        if isinstance(tree, dict):
            for key, value in tree.items():
                current_path = f"{path}.{key}" if path else key
                if 'GLU_0' in current_path and 'Dense_0' in current_path and 'kernel' in current_path:
                    return value
                result = find_target_layer(value, current_path)
                if result is not None:
                    return result
        elif hasattr(tree, 'items'):  # Handle other dict-like objects
            for key, value in tree.items():
                current_path = f"{path}.{key}" if path else str(key)
                if 'GLU_0' in current_path and 'Dense_0' in current_path and 'kernel' in current_path:
                    return value
                result = find_target_layer(value, current_path)
                if result is not None:
                    return result
        return None
    
    grad_tensor_train = find_target_layer(grads_train)
    grad_tensor_inference = find_target_layer(grads_inference)
    
    if grad_tensor_train is not None and grad_tensor_inference is not None:
        # Compute influences for both
        def compute_influences(tensor):
            if tensor.ndim == 3:
                axes_to_norm = tuple(range(tensor.ndim - 1))
                return jnp.linalg.norm(tensor, axis=axes_to_norm)
            elif tensor.ndim == 2:
                return jnp.linalg.norm(tensor, axis=0)
            else:
                return jnp.abs(tensor)
        
        influences_train = compute_influences(grad_tensor_train)
        influences_inference = compute_influences(grad_tensor_inference)
        
        print(f"\n📊 Results for GLU_0.Dense_0.kernel:")
        print(f"   🏋️ Train Mode:")
        print(f"      Max: {float(jnp.max(influences_train)):.6f}")
        print(f"      Mean: {float(jnp.mean(influences_train)):.6f}")
        print(f"   🔍 Inference Mode:")
        print(f"      Max: {float(jnp.max(influences_inference)):.6f}")
        print(f"      Mean: {float(jnp.mean(influences_inference)):.6f}")
        
        # Compute ratio
        ratio_max = float(jnp.max(influences_train)) / float(jnp.max(influences_inference))
        ratio_mean = float(jnp.mean(influences_train)) / float(jnp.mean(influences_inference))
        
        print(f"   📈 Ratios:")
        print(f"      Max ratio (train/inference): {ratio_max:.2f}x")
        print(f"      Mean ratio (train/inference): {ratio_mean:.2f}x")
        
        return {
            'train_max': float(jnp.max(influences_train)),
            'train_mean': float(jnp.mean(influences_train)),
            'inference_max': float(jnp.max(influences_inference)),
            'inference_mean': float(jnp.mean(influences_inference)),
            'ratio_max': ratio_max,
            'ratio_mean': ratio_mean
        }
    else:
        print("❌ Could not find target layer in gradients")
    
    return None

def main():
    print("🔍 DEBUGGING MAGNITUDE DISCREPANCY")
    print("=" * 50)
    
    # Load training influences
    print("🔄 Loading training influence data...")
    training_data = load_training_influences()
    
    if training_data:
        print("✅ Found training data for image 22951:")
        layer_influences = training_data.get('layer_influences', {})
        for layer_name, influences in layer_influences.items():
            if 'GLU_0.Dense_0.kernel' in layer_name:
                if isinstance(influences, list):
                    print(f"   📊 {layer_name}: {len(influences)} neurons")
                    print(f"      Max: {max(influences):.6f}, Mean: {np.mean(influences):.6f}")
                else:
                    print(f"   📊 {layer_name}: {influences:.6f} (scalar)")
    else:
        print("❌ No training data found")
        return
    
    # Load model and image
    print("\n🔄 Loading model and image...")
    model_path = "outputs/dalle-mini-from-scratch"
    model = DalleBart.from_pretrained(model_path, local_files_only=True)
    tokenizer = DalleBartTokenizer.from_pretrained(model_path, local_files_only=True)
    
    image_data = load_image_22951()
    if not image_data:
        print("❌ Image 22951 not found")
        return
    
    original_encoding = image_data['encoding']
    print(f"✅ Loaded image 22951: {len(original_encoding)} tokens")
    
    # Test 1: Different sequence lengths
    length_results = test_different_sequence_lengths(model, tokenizer, original_encoding)
    
    # Test 2: Train vs Inference mode
    mode_results = compare_train_vs_inference_mode(model, original_encoding)
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY AND CONCLUSIONS")
    print("=" * 50)
    
    print("\n🎯 Key Findings:")
    
    if length_results:
        print("\n📏 Sequence Length Impact:")
        for length, result in length_results.items():
            print(f"   Length {length}: Max={result['max']:.6f}, Mean={result['mean']:.6f}")
    
    if mode_results:
        print(f"\n🔄 Train vs Inference Mode:")
        print(f"   Train mode produces {mode_results['ratio_max']:.1f}x higher max influences")
        print(f"   Train mode produces {mode_results['ratio_mean']:.1f}x higher mean influences")
    
    print("\n💡 Recommendations:")
    print("1. If train/inference mode difference is significant, use train=True in post-training")
    print("2. If sequence length matters, use full 256 tokens (if NaN issue can be resolved)")
    print("3. Consider if training context (batch, accumulation) affects gradient computation")

if __name__ == "__main__":
    main()
