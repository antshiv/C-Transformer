#!/usr/bin/env python3
"""
Simple GPT-2 Token Decoder
Extracts token IDs from output and decodes them.
Usage: python3 decode.py output.txt
"""
import sys
import re

def load_tokenizer():
    """Load GPT-2 tokenizer with error handling."""
    try:
        from transformers import GPT2Tokenizer
        return GPT2Tokenizer.from_pretrained('gpt2')
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        print("\n💡 Try: pip3 install transformers --break-system-packages")
        sys.exit(1)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 decode.py output.txt")
        print("\nExtracts 'Generated token ID: XXX' lines and decodes them")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    # Extract tokens from file
    tokens = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                match = re.search(r'Generated token ID:\s*(\d+)', line)
                if match:
                    tokens.append(int(match.group(1)))
    except FileNotFoundError:
        print(f"❌ File not found: {filename}")
        sys.exit(1)
    
    if not tokens:
        print(f"❌ No tokens found in {filename}")
        print("   Looking for lines like: 'Generated token ID: 262'")
        sys.exit(1)
    
    print(f"📖 Reading: {filename}")
    print(f"✅ Found {len(tokens)} tokens")
    print()
    
    # Load tokenizer
    print("🔄 Loading tokenizer...")
    tokenizer = load_tokenizer()
    
    # Decode
    decoded_text = tokenizer.decode(tokens)
    
    # Show results
    print("=" * 70)
    print("🎯 RESULTS")
    print("=" * 70)
    print(f"\nToken IDs: {tokens}")
    print(f"\nDecoded text:")
    print(f"  {decoded_text}")
    print()
    print("=" * 70)
    
    # Show individual tokens
    print("\n🔍 Token breakdown:")
    for i, token_id in enumerate(tokens):
        decoded = tokenizer.decode([token_id])
        print(f"  [{i:2d}] {token_id:5d} → '{decoded}'")
    print()

if __name__ == "__main__":
    main()
