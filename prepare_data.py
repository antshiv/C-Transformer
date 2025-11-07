#!/usr/bin/env python3
"""
Prepare GPT-2 training data from TinyStories dataset.
"""

from datasets import load_dataset
import pandas as pd  # <-- Make sure this is here
import tiktoken
from pathlib import Path
import struct
import json

def main():
    print("="*60)
    print("GPT-2 Training Data Preparation")
    print("="*60)
    
    # Load dataset
    print("\n1. Loading TinyStories dataset...")
    ds = load_dataset("roneneldan/TinyStories")
    df = ds['train'].to_pandas()
    print(f"   Loaded {len(df)} stories")
    
    # Pack stories
    print("\n2. Packing stories to target token count...")
    tokenizer = tiktoken.get_encoding("gpt2")
    target_tokens = 20000
    
    packed_stories = []
    current_tokens = 0
    
    for idx, story in enumerate(df['text']):
        packed_stories.append(story)
        current_tokens = len(tokenizer.encode("\n\n".join(packed_stories)))
        
        if current_tokens >= target_tokens:
            break
    
    packed_text = "\n\n".join(packed_stories)
    tokens = tokenizer.encode(packed_text)
    
    print(f"   Packed {len(packed_stories)} stories")
    print(f"   Total tokens: {len(tokens)}")
    print(f"   Training pairs: {len(tokens) - 1024}")
    
    # Create training pairs
    print("\n3. Creating training pairs...")
    context_length = 1024
    
    training_pairs = []
    for i in range(len(tokens) - context_length):
        sequence = tokens[i:i+context_length+1]  # 1025 tokens
        training_pairs.append({
            'pair_id': i,
            'tokens': sequence
        })
    
    df_pairs = pd.DataFrame(training_pairs)
    print(f"   Created {len(df_pairs)} training pairs")
    
    # Save binary files
    print("\n4. Saving training pairs as binary files...")
    pairs_dir = Path("./data/training_pairs")
    pairs_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, row in df_pairs.iterrows():
        sequence = row['tokens']
        pair_file = pairs_dir / f"pair_{idx:05d}.bin"
        
        with open(pair_file, 'wb') as f:
            f.write(struct.pack(f'{len(sequence)}I', *sequence))
        
        if idx % 1000 == 0:
            print(f"   Saved {idx}/{len(df_pairs)}")
    
    print(f"   ✓ Saved {len(df_pairs)} binary files to {pairs_dir}")
    
    # Save metadata
    print("\n5. Saving metadata...")
    metadata = {
        'num_stories': len(packed_stories),
        'num_tokens': len(tokens),
        'num_pairs': len(df_pairs),
        'context_length': context_length,
        'vocab_size': 50257,
        'tokenizer': 'gpt2'
    }
    
    metadata_file = pairs_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   ✓ Saved metadata to {metadata_file}")
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING DATA READY")
    print("="*60)
    print(f"Total stories packed:  {metadata['num_stories']}")
    print(f"Total tokens:          {metadata['num_tokens']}")
    print(f"Training pairs:        {metadata['num_pairs']}")
    print(f"Context length:        {metadata['context_length']}")
    print(f"Tokens per file:       1025")
    print(f"Files location:        {pairs_dir}")
    print("\nReady to train!")

if __name__ == "__main__":
    main()
