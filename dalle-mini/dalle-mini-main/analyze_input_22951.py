#!/usr/bin/env python3
"""
Analyze Input 22951 Script
==========================
This script analyzes the specific input tokens for image 22951 to understand
why they cause NaN in the model output when simple inputs work fine.
"""

import os
import jax
import jax.numpy as jnp
import numpy as np
from datasets import load_dataset
from dalle_mini.model import DalleBart, DalleBartTokenizer

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

def analyze_tokens(tokens, tokenizer):
    """Analyze the token sequence for potential issues."""
    print(f"🔍 Token Analysis:")
    print(f"   Shape: {tokens.shape}")
    print(f"   Dtype: {tokens.dtype}")
    print(f"   Range: [{jnp.min(tokens)}, {jnp.max(tokens)}]")
    print(f"   Unique tokens: {len(jnp.unique(tokens))}")
    
    # Check for invalid token IDs
    vocab_size = tokenizer.vocab_size
    print(f"   Vocab size: {vocab_size}")
    
    invalid_tokens = tokens >= vocab_size
    if jnp.any(invalid_tokens):
        print(f"   ⚠️ Found {jnp.sum(invalid_tokens)} invalid tokens (>= vocab_size)")
        invalid_indices = jnp.where(invalid_tokens)[0]
        print(f"   Invalid token positions: {invalid_indices[:10]}...")  # Show first 10
        print(f"   Invalid token values: {tokens[invalid_indices][:10]}...")
    else:
        print(f"   ✅ All tokens are within valid range [0, {vocab_size-1}]")
    
    # Check for special patterns
    zero_tokens = jnp.sum(tokens == 0)
    pad_tokens = jnp.sum(tokens == tokenizer.pad_token_id) if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None else 0
    
    print(f"   Zero tokens: {zero_tokens}")
    print(f"   Pad tokens: {pad_tokens}")
    
    # Show token distribution
    unique_tokens, counts = jnp.unique(tokens, return_counts=True)
    print(f"   Most common tokens:")
    sorted_indices = jnp.argsort(counts)[::-1]
    for i in range(min(5, len(unique_tokens))):
        idx = sorted_indices[i]
        token_id = unique_tokens[idx]
        count = counts[idx]
        print(f"      Token {token_id}: {count} times")

def test_progressive_lengths(model, tokens, max_length=256):
    """Test the model with progressively longer sequences to find where it breaks."""
    print(f"\n🔍 Testing Progressive Lengths:")
    
    for length in [10, 20, 50, 100, 150, 200, 256]:
        if length > len(tokens[0]):
            break
            
        test_tokens = tokens[:, :length]
        print(f"\n   Testing length {length}:")
        print(f"   Input shape: {test_tokens.shape}")
        print(f"   Input range: [{jnp.min(test_tokens)}, {jnp.max(test_tokens)}]")
        
        try:
            outputs = model(input_ids=test_tokens, train=False)
            logits = outputs[0]
            
            has_nan = jnp.any(jnp.isnan(logits))
            has_inf = jnp.any(jnp.isinf(logits))
            
            if has_nan or has_inf:
                print(f"   ❌ Length {length}: NaN={has_nan}, Inf={has_inf}")
                print(f"   Logits range: [{float(jnp.min(logits)):.6f}, {float(jnp.max(logits)):.6f}]")
                return length  # Return the length where it breaks
            else:
                print(f"   ✅ Length {length}: Valid outputs")
                print(f"   Logits range: [{float(jnp.min(logits)):.6f}, {float(jnp.max(logits)):.6f}]")
                
        except Exception as e:
            print(f"   ❌ Length {length}: Error - {e}")
            return length
    
    return None

def test_token_subsets(model, tokens):
    """Test different subsets of tokens to isolate problematic ones."""
    print(f"\n🔍 Testing Token Subsets:")
    
    full_tokens = tokens[0]  # Remove batch dimension
    
    # Test first half
    print(f"\n   Testing first half (0-128):")
    first_half = full_tokens[:128].reshape(1, -1)
    try:
        outputs = model(input_ids=first_half, train=False)
        logits = outputs[0]
        has_issues = jnp.any(jnp.isnan(logits)) or jnp.any(jnp.isinf(logits))
        print(f"   {'❌' if has_issues else '✅'} First half: Issues={has_issues}")
    except Exception as e:
        print(f"   ❌ First half: Error - {e}")
    
    # Test second half
    print(f"\n   Testing second half (128-256):")
    second_half = full_tokens[128:].reshape(1, -1)
    try:
        outputs = model(input_ids=second_half, train=False)
        logits = outputs[0]
        has_issues = jnp.any(jnp.isnan(logits)) or jnp.any(jnp.isinf(logits))
        print(f"   {'❌' if has_issues else '✅'} Second half: Issues={has_issues}")
    except Exception as e:
        print(f"   ❌ Second half: Error - {e}")

def main():
    model_path = "outputs/dalle-mini-from-scratch"
    
    print("🔍 ANALYZING IMAGE 22951 INPUT")
    print("=" * 50)
    
    # Load model and tokenizer
    print("🔄 Loading model and tokenizer...")
    model = DalleBart.from_pretrained(model_path, local_files_only=True)
    tokenizer = DalleBartTokenizer.from_pretrained(model_path, local_files_only=True)
    
    # Load image 22951
    print("🔄 Loading image 22951...")
    example = load_image_22951()
    
    if not example:
        print("❌ Image 22951 not found!")
        return
    
    print(f"✅ Found image 22951")
    print(f"   Filename: {example['filename']}")
    print(f"   Caption: {example.get('caption', 'N/A')}")
    
    # Get tokens
    tokens = jnp.array([example['encoding']])  # Add batch dimension
    
    # Analyze tokens
    analyze_tokens(tokens, tokenizer)
    
    # Test progressive lengths
    break_length = test_progressive_lengths(model, tokens)
    
    if break_length:
        print(f"\n🎯 Model breaks at length: {break_length}")
        
        # Test token subsets to isolate the problem
        test_token_subsets(model, tokens)
    else:
        print(f"\n🤔 Model didn't break with progressive lengths - investigating further...")
    
    print("\n" + "=" * 50)
    print("RECOMMENDATIONS")
    print("=" * 50)
    
    if break_length:
        print(f"1. The model fails at sequence length {break_length}")
        print("2. This suggests either:")
        print("   - Specific problematic tokens in that range")
        print("   - Model instability with longer sequences")
        print("   - Position encoding issues")
        print("\n3. Potential solutions:")
        print("   - Use shorter sequences for influence computation")
        print("   - Filter out problematic tokens")
        print("   - Use a different model checkpoint")
    else:
        print("1. The issue might be more subtle")
        print("2. Consider checking:")
        print("   - Exact model configuration used during training")
        print("   - Different random seeds or initialization")
        print("   - Batch vs single example processing differences")

if __name__ == "__main__":
    main()
