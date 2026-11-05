#!/usr/bin/env python3
"""
Model Health Verification Script
================================
This script checks if the trained model has valid weights and can produce valid outputs.
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
from dalle_mini.model import DalleBart, DalleBartTokenizer

def check_model_weights(model_path):
    """Check if model weights contain NaN or infinite values."""
    print("🔍 Loading model...")
    model = DalleBart.from_pretrained(model_path, local_files_only=True)
    
    print("🔍 Checking model weights for NaN/Inf values...")
    
    def check_tensor(name, tensor):
        has_nan = jnp.any(jnp.isnan(tensor))
        has_inf = jnp.any(jnp.isinf(tensor))
        
        if has_nan or has_inf:
            print(f"⚠️ {name}: NaN={has_nan}, Inf={has_inf}, Shape={tensor.shape}")
            return True
        return False
    
    def check_params_recursive(params, prefix=""):
        issues_found = False
        if isinstance(params, dict):
            for key, value in params.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    issues_found |= check_params_recursive(value, full_key)
                else:
                    issues_found |= check_tensor(full_key, value)
        return issues_found
    
    issues_found = check_params_recursive(model.params)
    
    if issues_found:
        print("❌ Model weights contain NaN or Inf values!")
        return False
    else:
        print("✅ All model weights are finite and valid")
        return True

def test_model_forward(model_path):
    """Test model forward pass with simple input."""
    print("\n🔍 Testing model forward pass...")
    
    model = DalleBart.from_pretrained(model_path, local_files_only=True)
    tokenizer = DalleBartTokenizer.from_pretrained(model_path, local_files_only=True)
    
    # Create simple test input
    test_text = "a simple test"
    tokens = tokenizer.encode(test_text, return_tensors="jax", max_length=10, padding="max_length")
    
    print(f"📊 Test input shape: {tokens.shape}")
    print(f"📊 Test input range: [{jnp.min(tokens)}, {jnp.max(tokens)}]")
    
    try:
        # Forward pass
        outputs = model(input_ids=tokens, train=False)
        logits = outputs[0]
        
        print(f"📊 Output logits shape: {logits.shape}")
        print(f"📊 Logits range: [{float(jnp.min(logits)):.6f}, {float(jnp.max(logits)):.6f}]")
        
        # Check for issues
        has_nan = jnp.any(jnp.isnan(logits))
        has_inf = jnp.any(jnp.isinf(logits))
        
        if has_nan:
            print("❌ Model output contains NaN values!")
            return False
        elif has_inf:
            print("❌ Model output contains infinite values!")
            return False
        else:
            print("✅ Model forward pass produces valid outputs")
            return True
            
    except Exception as e:
        print(f"❌ Error in forward pass: {e}")
        return False

def analyze_layer_shapes(model_path):
    """Analyze the actual shapes of key layers."""
    print("\n🔍 Analyzing layer shapes...")
    
    model = DalleBart.from_pretrained(model_path, local_files_only=True)
    
    def find_layers_recursive(params, prefix=""):
        layers_info = {}
        if isinstance(params, dict):
            for key, value in params.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, dict):
                    layers_info.update(find_layers_recursive(value, full_key))
                else:
                    # Check if this is a key layer
                    if any(pattern in full_key for pattern in ['GLU_0.Dense_0.kernel', 'GLU_0.Dense_2.kernel', 'lm_head.kernel']):
                        layers_info[full_key] = {
                            'shape': value.shape,
                            'dtype': value.dtype,
                            'has_nan': bool(jnp.any(jnp.isnan(value))),
                            'has_inf': bool(jnp.any(jnp.isinf(value))),
                            'min': float(jnp.min(value)),
                            'max': float(jnp.max(value)),
                            'mean': float(jnp.mean(value))
                        }
        return layers_info
    
    layers_info = find_layers_recursive(model.params)
    
    print(f"🧠 Found {len(layers_info)} key layers:")
    for layer_name, info in layers_info.items():
        print(f"\n📊 {layer_name}:")
        print(f"   Shape: {info['shape']}")
        print(f"   Dtype: {info['dtype']}")
        print(f"   Range: [{info['min']:.6f}, {info['max']:.6f}]")
        print(f"   Mean: {info['mean']:.6f}")
        print(f"   Has NaN: {info['has_nan']}")
        print(f"   Has Inf: {info['has_inf']}")
    
    return layers_info

def main():
    model_path = "outputs/dalle-mini-from-scratch"
    
    print("🚀 DALL-E Mini Model Health Check")
    print("=" * 50)
    
    if not os.path.exists(model_path):
        print(f"❌ Model path does not exist: {model_path}")
        return
    
    # Check 1: Model weights
    weights_ok = check_model_weights(model_path)
    
    # Check 2: Forward pass
    forward_ok = test_model_forward(model_path)
    
    # Check 3: Layer analysis
    layers_info = analyze_layer_shapes(model_path)
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    if weights_ok and forward_ok:
        print("✅ Model appears healthy and ready for influence computation")
    else:
        print("❌ Model has issues that need to be resolved:")
        if not weights_ok:
            print("   - Model weights contain NaN/Inf values")
        if not forward_ok:
            print("   - Model forward pass produces invalid outputs")
        
        print("\n🔧 Possible solutions:")
        print("   1. Check if training completed successfully")
        print("   2. Verify model checkpoint integrity")
        print("   3. Use a different model checkpoint")
        print("   4. Retrain the model with better numerical stability")

if __name__ == "__main__":
    main()
