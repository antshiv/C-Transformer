#!/bin/bash
# Simple Doxygen documentation generator for C-Transformer

echo "🔧 Generating Doxygen documentation..."

# Check if doxygen is installed
if ! command -v doxygen &> /dev/null; then
    echo "❌ Error: doxygen is not installed"
    echo "Install with: sudo apt install doxygen graphviz"
    exit 1
fi

# Generate documentation
doxygen Doxyfile

# Check if generation was successful
if [ $? -eq 0 ]; then
    echo "✅ Documentation generated successfully!"
    echo "📂 Output directory: docs/html/"
    echo "🌐 Open docs/html/index.html in your browser"
    echo ""
    echo "To view locally, run:"
    echo "  firefox docs/html/index.html"
    echo "  # or"
    echo "  xdg-open docs/html/index.html"
else
    echo "❌ Error: Documentation generation failed"
    exit 1
fi
