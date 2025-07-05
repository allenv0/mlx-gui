#!/bin/bash

# MLX-GUI macOS App Builder
# This script builds a TRUE standalone macOS app bundle using PyInstaller

set -e

echo "🚀 Building MLX-GUI macOS App Bundle (TRUE STANDALONE)..."

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: pyproject.toml not found. Run this script from the project root."
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source .venv/bin/activate

# Install PyInstaller if not already installed
echo "📦 Installing PyInstaller..."
pip install pyinstaller

# Check for critical dependencies
echo "🔍 Checking critical dependencies..."
CRITICAL_DEPS=("mlx-lm" "mlx" "rumps" "fastapi" "uvicorn" "transformers" "huggingface-hub")
MISSING_DEPS=""

for dep in "${CRITICAL_DEPS[@]}"; do
    if ! pip show "$dep" > /dev/null 2>&1; then
        MISSING_DEPS="$MISSING_DEPS $dep"
    fi
done

if [ -n "$MISSING_DEPS" ]; then
    echo "❌ Missing critical dependencies:$MISSING_DEPS"
    echo "💡 Install with: pip install -e \".[app]\""
    echo "💡 Or from requirements: pip install -r requirements.txt"
    exit 1
fi

echo "✅ All critical dependencies found"

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf build/ dist/ *.spec

# Build the app using PyInstaller directly
echo "🔨 Building app bundle with PyInstaller..."

# Find MLX path for data files
MLX_PATH=$(pip show mlx | grep Location | cut -d ' ' -f 2)/mlx

pyinstaller src/mlx_gui/app_main.py \
    --name="MLX-GUI" \
    --onedir \
    --windowed \
    --noconfirm \
    --clean \
    --hidden-import=mlx \
    --hidden-import=mlx_lm \
    --hidden-import=mlx.core \
    --hidden-import=mlx.nn \
    --hidden-import=mlx.optimizers \
    --hidden-import=transformers \
    --hidden-import=tokenizers \
    --hidden-import=safetensors \
    --hidden-import=huggingface_hub \
    --hidden-import=fastapi \
    --hidden-import=uvicorn \
    --hidden-import=rumps \
    --hidden-import=objc \
    --hidden-import=AppKit \
    --hidden-import=Foundation \
    --hidden-import=CoreFoundation \
    --hidden-import=psutil \
    --hidden-import=sqlalchemy \
    --hidden-import=pydantic \
    --hidden-import=httpx \
    --hidden-import=requests \
    --hidden-import=typer \
    --hidden-import=rich \
    --hidden-import=PIL \
    --hidden-import=numpy \
    --hidden-import=sentencepiece \
    --hidden-import=protobuf \
    --hidden-import=regex \
    --hidden-import=yaml \
    --hidden-import=tqdm \
    --hidden-import=click \
    --hidden-import=aiofiles \
    --hidden-import=appdirs \
    --hidden-import=markdown_it_py \
    --hidden-import=jinja2 \
    --hidden-import=starlette \
    --hidden-import=uvloop \
    --hidden-import=websockets \
    --hidden-import=watchfiles \
    --hidden-import=python_multipart \
    --hidden-import=python_dotenv \
    --add-data="${MLX_PATH}:mlx" \
    --add-data="src/mlx_gui/templates:mlx_gui/templates" \
    --collect-all=mlx \
    --collect-all=mlx_lm \
    --collect-all=transformers \
    --collect-all=tokenizers \
    --collect-all=safetensors \
    --collect-all=huggingface_hub \
    --collect-all=rumps \
    --collect-all=objc \
    --target-arch=arm64 \
    --osx-bundle-identifier="org.matthewrogers.mlx-gui" \
    --log-level=INFO

# Check if build was successful
if [ -d "dist/MLX-GUI.app" ]; then
    echo "✅ App bundle built successfully!"
    echo "📍 Location: dist/MLX-GUI.app"
    echo ""
    echo "🎉 You can now:"
    echo "   1. Run: open dist/MLX-GUI.app"
    echo "   2. Copy to /Applications: cp -R dist/MLX-GUI.app /Applications/"
    echo "   3. Create a DMG installer"
    echo ""
    echo "📋 App Info:"
    echo "   - Size: $(du -sh dist/MLX-GUI.app | cut -f1)"
    echo "   - Type: TRUE STANDALONE (no Python required!)"
    echo "   - Includes: All Python runtime, MLX binaries, and dependencies"
    echo ""
    echo "🎯 This is a REAL standalone app!"
    echo "   - No Python installation required on target system"
    echo "   - No virtual environment needed"
    echo "   - Fully self-contained"
else
    echo "❌ Build failed! App bundle not found at dist/MLX-GUI.app"
    echo "   Check the output above for errors."
    exit 1
fi

echo ""
echo "🔗 Next steps:"
echo "   • Test the app: open dist/MLX-GUI.app"
echo "   • Create DMG installer for easy distribution"
echo "   • App is ready for sharing with anyone - no setup required!" 