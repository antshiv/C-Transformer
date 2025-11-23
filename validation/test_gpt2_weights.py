#!/usr/bin/env python3
"""
Test GPT-2 weights directly with PyTorch to verify they're correct.
This will tell us if the problem is in the weights or in the C code.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

print("="*60)
print("Testing GPT-2 Weights with PyTorch")
print("="*60)

# Load model and tokenizer
print("\n📥 Loading GPT-2 model from HuggingFace...")
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Test prompt (same as your C test)
test_prompt = "Once upon a time"

print(f"\n📝 Input: '{test_prompt}'")

# Tokenize
input_ids = tokenizer.encode(test_prompt, return_tensors='pt')
print(f"🔢 Tokens: {input_ids[0].tolist()}")

# Generate
print("\n🤖 Generating with PyTorch GPT-2...")
with torch.no_grad():
    output = model.generate(
        input_ids,
        max_length=24,  # 4 input + 20 generated = 24 total
        do_sample=False,  # Greedy decoding (like your C code)
        pad_token_id=tokenizer.eos_token_id
    )

# Decode
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("\n" + "="*60)
print("✅ PyTorch Generation Result:")
print("="*60)
print(generated_text)
print("="*60)

# Show token breakdown
print("\n🔍 Generated Tokens:")
output_tokens = output[0].tolist()
for i, token_id in enumerate(output_tokens):
    token_text = tokenizer.decode([token_id])
    marker = "📥 INPUT" if i < len(input_ids[0]) else "📤 GENERATED"
    print(f"  [{i:2d}] {token_id:5d} → {repr(token_text):30s} {marker}")

print("\n" + "="*60)
print("Comparison Instructions:")
print("="*60)
print("1. Compare this PyTorch output with your C code output")
print("2. If PyTorch output is good but C output is gibberish:")
print("   → Problem is in C code (forward pass, weight loading, etc.)")
print("3. If PyTorch output is also gibberish:")
print("   → Problem is in weight conversion (unlikely)")
print("4. If both outputs match exactly:")
print("   → Your C code is working correctly!")
print("="*60)
