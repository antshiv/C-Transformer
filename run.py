#!/usr/bin/env python3
"""
All-in-one GPT-2 runner: encode → run → decode
Usage: python3 run.py "Your text here"
"""
import sys
import subprocess
import re
import os
from datetime import datetime

def load_tokenizer():
    """Load GPT-2 tokenizer."""
    try:
        from transformers import GPT2Tokenizer
        return GPT2Tokenizer.from_pretrained('gpt2')
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        print("💡 Try: pip3 install transformers --break-system-packages")
        sys.exit(1)

def main():
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python3 run.py \"Your text here\" [options]")
        print("\nOptions:")
        print("  --weights FILE    Weights file (default: gpt2_bump.weights)")
        print("  --executable FILE C executable (default: ./main)")
        print("  --output FILE     Save output to file (default: auto-generated)")
        print("  --num-tokens N    Number of tokens to generate (default: 20)")
        print("\nExamples:")
        print('  python3 run.py "Hello, I am"')
        print('  python3 run.py "What is AI?" --weights gpt2.weights')
        print('  python3 run.py "The future" --output result.txt')
        sys.exit(1)
    
    # Get text (everything until first --)
    text_parts = []
    i = 1
    while i < len(sys.argv) and not sys.argv[i].startswith('--'):
        text_parts.append(sys.argv[i])
        i += 1
    text = ' '.join(text_parts)
    
    # Parse options
    weights = "gpt2_bump.weights"
    executable = "./main"
    output_file = None
    num_tokens = 20
    
    while i < len(sys.argv):
        if sys.argv[i] == '--weights' and i + 1 < len(sys.argv):
            weights = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--executable' and i + 1 < len(sys.argv):
            executable = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--output' and i + 1 < len(sys.argv):
            output_file = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--num-tokens' and i + 1 < len(sys.argv):
            num_tokens = int(sys.argv[i + 1])
            i += 2
        else:
            i += 1
    
    # Auto-generate output filename if not specified
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"output_{timestamp}.txt"
    
    print("=" * 70)
    print("🚀 GPT-2 All-in-One Runner")
    print("=" * 70)
    print(f"📝 Input text: {text}")
    print(f"⚙️  Weights: {weights}")
    print(f"🎯 Generate: {num_tokens} tokens")
    print(f"💾 Output: {output_file}")
    print()
    
    # Check if executable exists
    if not os.path.exists(executable):
        print(f"❌ Executable not found: {executable}")
        print("💡 Compile your C code first:")
        print(f"   gcc -O3 -march=native -mavx512f -fopenmp main.c -o main -lm")
        sys.exit(1)
    
    # Check if weights exist
    if not os.path.exists(weights):
        print(f"❌ Weights file not found: {weights}")
        sys.exit(1)
    
    # STEP 1: Encode
    print("🔄 Step 1: Encoding text to tokens...")
    tokenizer = load_tokenizer()
    tokens = tokenizer.encode(text)
    tokens_str = ','.join(map(str, tokens))
    
    print(f"✅ Encoded to {len(tokens)} tokens: {tokens}")
    print()
    
    # STEP 2: Run C program
    print("🔄 Step 2: Running C program...")
    cmd = [
        executable,
        '--weights', weights,
        '--prompt', tokens_str,
        '--force'
    ]
    print(f"   Command: {' '.join(cmd)}")
    print()
    
    try:
        # Run and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        # Save full output
        with open(output_file, 'w') as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n=== STDERR ===\n")
                f.write(result.stderr)
        
        print(f"✅ C program completed")
        print(f"💾 Full output saved to: {output_file}")
        print()
        
        # STEP 3: Extract and decode tokens
        print("🔄 Step 3: Extracting generated tokens...")
        generated_tokens = []
        for line in result.stdout.split('\n'):
            match = re.search(r'Generated token ID:\s*(\d+)', line)
            if match:
                generated_tokens.append(int(match.group(1)))
        
        if not generated_tokens:
            print("⚠️  No generated tokens found in output")
            print(f"   Check {output_file} for details")
            sys.exit(1)
        
        print(f"✅ Found {len(generated_tokens)} generated tokens")
        print()
        
        # STEP 4: Decode
        print("🔄 Step 4: Decoding tokens to text...")
        
        # Combine input + generated tokens
        full_tokens = tokens + generated_tokens
        full_text = tokenizer.decode(full_tokens)
        generated_text = tokenizer.decode(generated_tokens)
        
        # Display results
        print("=" * 70)
        print("🎯 RESULTS")
        print("=" * 70)
        print()
        print(f"📥 INPUT:")
        print(f"   {text}")
        print()
        print(f"📤 GENERATED:")
        print(f"   {generated_text}")
        print()
        print(f"📖 FULL TEXT:")
        print(f"   {full_text}")
        print()
        print("=" * 70)
        
        # Show token breakdown
        print()
        print("🔍 Generated Token Breakdown:")
        for i, token_id in enumerate(generated_tokens):
            decoded = tokenizer.decode([token_id])
            print(f"  [{i:2d}] {token_id:5d} → '{decoded}'")
        print()
        
        print(f"💾 Full output saved to: {output_file}")
        print("=" * 70)
        
    except subprocess.TimeoutExpired:
        print("❌ Timeout: C program took too long (>5 minutes)")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
