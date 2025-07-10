#!/bin/bash

# MLX-GUI Multimodal Vision Test Runner
# This script runs the multimodal vision test for the Gemma-3 model

echo "🚀 MLX-GUI Multimodal Vision Test Runner"
echo "=========================================="

# Check if we're in the right directory
if [[ ! -f "icon.png" ]]; then
    echo "❌ Error: icon.png not found. Please run this from the mlx-gui project root."
    exit 1
fi

# Check if the MLX-GUI server is running
echo "🔍 Checking if MLX-GUI server is running..."
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "❌ MLX-GUI server is not running!"
    echo "   Please start it first:"
    echo "   mlx-gui start --port 8000"
    echo ""
    echo "   Or if you're in development mode:"
    echo "   python -m mlx_gui.cli start --port 8000"
    exit 1
fi

echo "✅ Server is running"

# Run the test
echo "🧪 Running multimodal vision test..."
echo ""

cd "$(dirname "$0")/.." || exit 1
python tests/test_multimodal_vision.py

exit_code=$?

echo ""
if [[ $exit_code -eq 0 ]]; then
    echo "🎉 Test completed successfully!"
    echo ""
    echo "📋 Summary:"
    echo "   ✅ Gemma-3 vision model installed and loaded"
    echo "   ✅ Image processing working"
    echo "   ✅ Multimodal chat completions functional"
    echo "   ✅ Model can see the 'M' in icon.png"
else
    echo "💥 Test failed!"
    echo ""
    echo "🔧 Troubleshooting:"
    echo "   1. Make sure MLX-GUI server is running: mlx-gui start"
    echo "   2. Check server logs for errors"
    echo "   3. Verify mlx-vlm is installed: pip install mlx-vlm"
    echo "   4. Ensure you have sufficient memory for the model"
fi

exit $exit_code