#!/usr/bin/env python3
"""
Debug script to verify C weight loading matches PyTorch weights.
We'll generate gpt2_bump.weights and then read it back to verify.
"""

import struct
import numpy as np
from transformers import GPT2LMHeadModel

print("="*70)
print("WEIGHT VERIFICATION TEST")
print("="*70)

# Load PyTorch model
print("\n1. Loading GPT-2 from HuggingFace...")
model = GPT2LMHeadModel.from_pretrained('gpt2')
state_dict = model.state_dict()

# Check if we're using GPT2Model or GPT2LMHeadModel
if 'transformer.wte.weight' in state_dict:
    print("   ✓ Using GPT2LMHeadModel (keys have 'transformer.' prefix)")
    wte_key = 'transformer.wte.weight'
    wpe_key = 'transformer.wpe.weight'
    ln_key = 'transformer.h.0.ln_1.weight'
else:
    print("   ✓ Using GPT2Model (keys have no prefix)")
    wte_key = 'wte.weight'
    wpe_key = 'wpe.weight'
    ln_key = 'h.0.ln_1.weight'

# Get reference weights
wte_pytorch = state_dict[wte_key].numpy()
wpe_pytorch = state_dict[wpe_key].numpy()
ln1_weight_pytorch = state_dict[ln_key].numpy()

print(f"\n2. PyTorch Reference Weights:")
print(f"   Token embeddings shape: {wte_pytorch.shape}")
print(f"   First 10 values: {wte_pytorch[0, :10]}")
print(f"   Position embeddings shape: {wpe_pytorch.shape}")
print(f"   Layer 0 LN1 weight shape: {ln1_weight_pytorch.shape}")
print(f"   Layer 0 LN1 first 10: {ln1_weight_pytorch[:10]}")

# Check if weights file exists
import os
if not os.path.exists('gpt2_bump.weights'):
    print("\n⚠️  WARNING: gpt2_bump.weights not found!")
    print("   Run the pytorch_to_c_weights.ipynb notebook first to generate it.")
    exit(1)

print("\n3. Reading C weight file (gpt2_bump.weights)...")

# Read header
with open('gpt2_bump.weights', 'rb') as f:
    magic = f.read(8)
    version = struct.unpack('I', f.read(4))[0]
    model_type = struct.unpack('I', f.read(4))[0]

    # Hyperparameters
    n_layers = struct.unpack('I', f.read(4))[0]
    vocab_size = struct.unpack('I', f.read(4))[0]
    embed_dim = struct.unpack('I', f.read(4))[0]
    context_len = struct.unpack('I', f.read(4))[0]
    n_heads = struct.unpack('I', f.read(4))[0]
    head_dim = struct.unpack('I', f.read(4))[0]

    # Aligned dimensions
    aligned_embed_dim = struct.unpack('Q', f.read(8))[0]
    aligned_head_dim = struct.unpack('Q', f.read(8))[0]
    aligned_context = struct.unpack('Q', f.read(8))[0]

    print(f"   Header info:")
    print(f"     Magic: {magic}")
    print(f"     Version: {version}")
    print(f"     Layers: {n_layers}, Vocab: {vocab_size}, Embed: {embed_dim}")
    print(f"     Aligned embed: {aligned_embed_dim}")

    # Skip to weights (after 128-byte header)
    f.seek(128)

    # Read token embeddings
    print(f"\n4. Reading token embeddings from file...")
    wte_c = np.zeros((vocab_size, embed_dim), dtype=np.float32)
    for v in range(vocab_size):
        row = np.fromfile(f, dtype=np.float32, count=aligned_embed_dim)
        wte_c[v] = row[:embed_dim]  # Strip padding

    print(f"   Token embeddings shape: {wte_c.shape}")
    print(f"   First 10 values: {wte_c[0, :10]}")

    # Read position embeddings
    print(f"\n5. Reading position embeddings from file...")
    wpe_c = np.zeros((context_len, embed_dim), dtype=np.float32)
    for pos in range(context_len):
        row = np.fromfile(f, dtype=np.float32, count=aligned_embed_dim)
        wpe_c[pos] = row[:embed_dim]

    print(f"   Position embeddings shape: {wpe_c.shape}")

    # Read layer 0 ln1 weight
    print(f"\n6. Reading layer 0 LayerNorm1 weight...")
    ln1_weight_c = np.fromfile(f, dtype=np.float32, count=aligned_embed_dim)[:embed_dim]
    print(f"   Layer 0 LN1 weight shape: {ln1_weight_c.shape}")
    print(f"   First 10 values: {ln1_weight_c[:10]}")

# Compare
print("\n" + "="*70)
print("COMPARISON RESULTS")
print("="*70)

# Token embeddings
wte_diff = np.abs(wte_pytorch - wte_c).max()
wte_match = np.allclose(wte_pytorch, wte_c, rtol=1e-5, atol=1e-6)
print(f"\n1. Token Embeddings:")
print(f"   Max difference: {wte_diff:.2e}")
print(f"   Match: {'✓ YES' if wte_match else '✗ NO'}")
if not wte_match:
    print(f"   ⚠️  MISMATCH DETECTED!")
    print(f"   PyTorch [0,:5]: {wte_pytorch[0, :5]}")
    print(f"   C file  [0,:5]: {wte_c[0, :5]}")

# Position embeddings
wpe_diff = np.abs(wpe_pytorch - wpe_c).max()
wpe_match = np.allclose(wpe_pytorch, wpe_c, rtol=1e-5, atol=1e-6)
print(f"\n2. Position Embeddings:")
print(f"   Max difference: {wpe_diff:.2e}")
print(f"   Match: {'✓ YES' if wpe_match else '✗ NO'}")
if not wpe_match:
    print(f"   ⚠️  MISMATCH DETECTED!")

# LayerNorm
ln_diff = np.abs(ln1_weight_pytorch - ln1_weight_c).max()
ln_match = np.allclose(ln1_weight_pytorch, ln1_weight_c, rtol=1e-5, atol=1e-6)
print(f"\n3. Layer 0 LayerNorm1 Weight:")
print(f"   Max difference: {ln_diff:.2e}")
print(f"   Match: {'✓ YES' if ln_match else '✗ NO'}")
if not ln_match:
    print(f"   ⚠️  MISMATCH DETECTED!")
    print(f"   PyTorch [:5]: {ln1_weight_pytorch[:5]}")
    print(f"   C file  [:5]: {ln1_weight_c[:5]}")

print("\n" + "="*70)
if wte_match and wpe_match and ln_match:
    print("✅ WEIGHTS ARE CORRECT!")
    print("   The bug is likely in your C forward pass, not weight loading.")
else:
    print("❌ WEIGHTS MISMATCH!")
    print("   The notebook weight conversion has a bug.")
print("="*70)
