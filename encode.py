#!/usr/bin/env python3
"""
Simple GPT-2 Token Encoder
Usage: python3 encode.py "Your text here"
"""
import sys

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
        print("Usage: python3 encode.py \"Your text here\"")
        print("\nExamples:")
        print('  python3 encode.py "Hello, I am"')
        print('  python3 encode.py "What is the capital of France?"')
        sys.exit(1)
    
    # Get text from command line
    text = ' '.join(sys.argv[1:])
    
    print(f"📝 Input text: {text}")
    print()
    
    # Load tokenizer
    print("🔄 Loading tokenizer...")
    tokenizer = load_tokenizer()
    
    # Encode
    tokens = tokenizer.encode(text)
    
    # Show results
    print("=" * 70)
    print(f"✅ Encoded {len(tokens)} tokens")
    print("=" * 70)
    
    # Format for C array
    tokens_str = ','.join(map(str, tokens))
    print(f"\n📋 C array format:")
    print(f"int prompt[] = {{{tokens_str}}};")
    print(f"int prompt_length = {len(tokens)};")
    
    # Format for command line
    print(f"\n🚀 Command line format:")
    print(f'./main --weights gpt2_bump.weights --prompt "{tokens_str}" --force')
    
    # Show individual tokens
    print(f"\n🔍 Token breakdown:")
    for i, token_id in enumerate(tokens):
        decoded = tokenizer.decode([token_id])
        print(f"  [{i:2d}] {token_id:5d} → '{decoded}'")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
