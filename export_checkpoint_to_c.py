#!/usr/bin/env python3
"""
Export a fine-tuned GPT-2 checkpoint to C-compatible binary format.

Usage:
  # Export specific checkpoint
  python3 export_checkpoint_to_c.py --checkpoint checkpoints/sql_gpt2/checkpoint-50 --output sql_step50.weights

  # Export final model
  python3 export_checkpoint_to_c.py --checkpoint checkpoints/sql_gpt2/final --output sql_final.weights
"""

import argparse
import os
import torch
import numpy as np
import struct
import hashlib
from transformers import GPT2LMHeadModel

# Match C alignment constants
CACHE_ALIGN = 64
FLOAT_SIZE = 4

def align_up(n, alignment=CACHE_ALIGN):
    """Align number of floats to cache boundary"""
    bytes_needed = n * FLOAT_SIZE
    aligned_bytes = ((bytes_needed + alignment - 1) // alignment) * alignment
    return aligned_bytes // FLOAT_SIZE


def export_checkpoint_to_c_weights(checkpoint_dir, output_file):
    """
    Load a HuggingFace GPT2LMHeadModel checkpoint and export to C binary format.
    """
    if not os.path.isdir(checkpoint_dir):
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return False

    print(f"📂 Loading checkpoint from {checkpoint_dir}...")
    try:
        model = GPT2LMHeadModel.from_pretrained(checkpoint_dir)
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return False

    state_dict = model.state_dict()
    config = model.config

    # Extract config
    n_layers = config.n_layer
    embed_dim = config.n_embd
    vocab_size = config.vocab_size
    context_len = config.n_positions
    n_heads = config.n_head
    head_dim = embed_dim // n_heads

    aligned_embed_dim = align_up(embed_dim)
    aligned_head_dim = align_up(head_dim)
    aligned_context = align_up(context_len)

    print(f"📊 Model config:")
    print(f"   Layers:      {n_layers}")
    print(f"   Embed dim:   {embed_dim} (aligned: {aligned_embed_dim})")
    print(f"   Vocab size:  {vocab_size}")
    print(f"   Context:     {context_len}")
    print(f"   Heads:       {n_heads}")
    print(f"   Head dim:    {head_dim} (aligned: {aligned_head_dim})")

    # GPT2LMHeadModel uses 'transformer.' prefix
    key_prefix = 'transformer.'

    with open(output_file, 'wb') as f:
        # Reserve header space
        header_size = 128
        f.write(b'\x00' * header_size)
        current_offset = header_size

        def write_aligned(tensor, name, expected_floats):
            nonlocal current_offset
            data = tensor.detach().cpu().numpy().astype(np.float32)
            flat = data.flatten()
            actual = len(flat)

            # Create aligned buffer
            aligned_buffer = np.zeros(expected_floats, dtype=np.float32)
            aligned_buffer[:actual] = flat

            bytes_written = f.write(aligned_buffer.tobytes())
            current_offset += bytes_written
            print(f"  {name:20s}: {str(data.shape):20s} -> {expected_floats:8d} floats")
            return bytes_written

        print("\n📝 Writing weights...")

        # Token embeddings
        write_aligned(state_dict[f'{key_prefix}wte.weight'], 'token_emb',
                      vocab_size * aligned_embed_dim)

        # Position embeddings
        write_aligned(state_dict[f'{key_prefix}wpe.weight'], 'pos_emb',
                      context_len * aligned_embed_dim)

        # Layers
        for layer_idx in range(n_layers):
            prefix = f'{key_prefix}h.{layer_idx}'
            print(f"\n  Layer {layer_idx}:")

            # LayerNorm 1
            write_aligned(state_dict[f'{prefix}.ln_1.weight'], 'ln1_gamma', aligned_embed_dim)
            write_aligned(state_dict[f'{prefix}.ln_1.bias'], 'ln1_beta', aligned_embed_dim)

            # QKV split (HuggingFace stores as concatenated [D, 3*D])
            qkv_w = state_dict[f'{prefix}.attn.c_attn.weight'].T  # [3*D, D] -> [D, 3*D]
            qkv_b = state_dict[f'{prefix}.attn.c_attn.bias']      # [3*D]

            q_w = qkv_w[:embed_dim, :]
            k_w = qkv_w[embed_dim:2*embed_dim, :]
            v_w = qkv_w[2*embed_dim:3*embed_dim, :]

            q_b = qkv_b[:embed_dim]
            k_b = qkv_b[embed_dim:2*embed_dim]
            v_b = qkv_b[2*embed_dim:3*embed_dim]

            write_aligned(q_w, 'q_weight', aligned_embed_dim * aligned_embed_dim)
            write_aligned(q_b, 'q_bias', aligned_embed_dim)
            write_aligned(k_w, 'k_weight', aligned_embed_dim * aligned_embed_dim)
            write_aligned(k_b, 'k_bias', aligned_embed_dim)
            write_aligned(v_w, 'v_weight', aligned_embed_dim * aligned_embed_dim)
            write_aligned(v_b, 'v_bias', aligned_embed_dim)

            # Attention projection
            proj_w = state_dict[f'{prefix}.attn.c_proj.weight'].T  # [D, D] -> [D, D]
            proj_b = state_dict[f'{prefix}.attn.c_proj.bias']      # [D]
            write_aligned(proj_w, 'proj_weight', aligned_embed_dim * aligned_embed_dim)
            write_aligned(proj_b, 'proj_bias', aligned_embed_dim)

            # LayerNorm 2
            write_aligned(state_dict[f'{prefix}.ln_2.weight'], 'ln2_gamma', aligned_embed_dim)
            write_aligned(state_dict[f'{prefix}.ln_2.bias'], 'ln2_beta', aligned_embed_dim)

            # MLP
            fc1_w = state_dict[f'{prefix}.mlp.c_fc.weight'].T     # [D, 4*D] -> [D, 4*D]
            fc1_b = state_dict[f'{prefix}.mlp.c_fc.bias']         # [4*D]
            write_aligned(fc1_w, 'fc1_weight', 4 * aligned_embed_dim * aligned_embed_dim)
            write_aligned(fc1_b, 'fc1_bias', 4 * aligned_embed_dim)

            fc2_w = state_dict[f'{prefix}.mlp.c_proj.weight'].T   # [4*D, D] -> [4*D, D]
            fc2_b = state_dict[f'{prefix}.mlp.c_proj.bias']       # [D]
            write_aligned(fc2_w, 'fc2_weight', 4 * aligned_embed_dim * aligned_embed_dim)
            write_aligned(fc2_b, 'fc2_bias', aligned_embed_dim)

        # Final LayerNorm
        print("\n  Final LayerNorm:")
        write_aligned(state_dict[f'{key_prefix}ln_f.weight'], 'ln_f_gamma', aligned_embed_dim)
        write_aligned(state_dict[f'{key_prefix}ln_f.bias'], 'ln_f_beta', aligned_embed_dim)

    # Write header with checksum
    with open(output_file, 'r+b') as f:
        f.seek(128)
        weight_data = f.read()
        checksum = hashlib.sha256(weight_data).digest()

        f.seek(0)
        f.write(b'BUMPWGT2')                           # Magic (8 bytes)
        f.write(struct.pack('I', 2))                   # Version
        f.write(struct.pack('I', 0))                   # Reserved
        f.write(struct.pack('I', n_layers))            # n_layer
        f.write(struct.pack('I', vocab_size))          # vocab_size
        f.write(struct.pack('I', embed_dim))           # embed_dim
        f.write(struct.pack('I', context_len))         # context_len
        f.write(struct.pack('I', n_heads))             # n_heads
        f.write(struct.pack('I', head_dim))            # head_dim
        f.write(struct.pack('Q', aligned_embed_dim))   # aligned_embed_dim
        f.write(struct.pack('Q', aligned_head_dim))    # aligned_head_dim
        f.write(struct.pack('Q', aligned_context))     # aligned_context
        f.write(checksum)                              # SHA256 checksum (32 bytes)
        f.write(b'\x00' * 32)                          # Reserved (32 bytes)

    print(f"\n✅ Successfully exported to {output_file}")
    print(f"   File size: {current_offset / (1024**2):.2f} MB")
    print(f"   Checksum:  {checksum.hex()[:16]}...")

    # Load and show training metadata if available
    import json
    metadata_file = os.path.join(checkpoint_dir, "training_info.json")
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        print(f"\n📋 Training metadata:")
        for key, value in metadata.items():
            print(f"   {key:15s}: {value}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Export HuggingFace GPT-2 checkpoint to C-compatible binary format"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint directory (e.g., checkpoints/sql_gpt2/checkpoint-50)"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output .weights file (e.g., sql_step50.weights)"
    )
    args = parser.parse_args()

    success = export_checkpoint_to_c_weights(args.checkpoint, args.output)

    if success:
        print(f"\n🚀 Ready to use with C inference:")
        print(f"   ./main --weights {args.output} --prompt \"[TASK: SQL] Question: What is...\" --force")
    else:
        exit(1)


if __name__ == "__main__":
    main()
