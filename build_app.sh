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
rm -rf build/ dist/ *.spec app_icon.icns

# Create app icon from PNG
echo "🎨 Creating app icon from ~/Downloads/icon.png..."
if [ -f ~/Downloads/icon.png ]; then
    # Create iconset directory
    mkdir -p app_icon.iconset
    
    # Generate different icon sizes using sips (built into macOS)
    sips -z 16 16 ~/Downloads/icon.png --out app_icon.iconset/icon_16x16.png
    sips -z 32 32 ~/Downloads/icon.png --out app_icon.iconset/icon_16x16@2x.png
    sips -z 32 32 ~/Downloads/icon.png --out app_icon.iconset/icon_32x32.png
    sips -z 64 64 ~/Downloads/icon.png --out app_icon.iconset/icon_32x32@2x.png
    sips -z 128 128 ~/Downloads/icon.png --out app_icon.iconset/icon_128x128.png
    sips -z 256 256 ~/Downloads/icon.png --out app_icon.iconset/icon_128x128@2x.png
    sips -z 256 256 ~/Downloads/icon.png --out app_icon.iconset/icon_256x256.png
    sips -z 512 512 ~/Downloads/icon.png --out app_icon.iconset/icon_256x256@2x.png
    sips -z 512 512 ~/Downloads/icon.png --out app_icon.iconset/icon_512x512.png
    sips -z 1024 1024 ~/Downloads/icon.png --out app_icon.iconset/icon_512x512@2x.png
    
    # Convert to icns format
    iconutil -c icns app_icon.iconset -o app_icon.icns
    
    # Clean up temporary iconset
    rm -rf app_icon.iconset
    
    echo "✅ App icon created: app_icon.icns"
else
    echo "⚠️  Warning: ~/Downloads/icon.png not found, using default icon"
fi

# Build the app using PyInstaller directly
echo "🔨 Building app bundle with PyInstaller..."

# Find MLX path for data files
MLX_PATH=$(pip show mlx | grep Location | cut -d ' ' -f 2)/mlx

# Check if we have a custom icon
ICON_PARAM=""
if [ -f "app_icon.icns" ]; then
    ICON_PARAM="--icon=app_icon.icns"
    echo "📱 Using custom app icon"
else
    echo "📱 Using default icon"
fi

pyinstaller src/mlx_gui/app_main.py \
    --name="MLX-GUI" \
    --onedir \
    --windowed \
    --noconfirm \
    --clean \
    $ICON_PARAM \
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

# Fix the Info.plist to make it a menu bar app (no dock icon)
echo "🔧 Converting to menu bar app (removing dock icon)..."
INFO_PLIST="dist/MLX-GUI.app/Contents/Info.plist"

if [ -f "$INFO_PLIST" ]; then
    # Add LSUIElement=true to make it a menu bar app
    /usr/libexec/PlistBuddy -c "Add :LSUIElement bool true" "$INFO_PLIST" 2>/dev/null || \
    /usr/libexec/PlistBuddy -c "Set :LSUIElement true" "$INFO_PLIST"
    
    echo "✅ App converted to menu bar app (no dock icon)"
    echo "   - App will only appear in the menu bar"
    echo "   - No dock icon will be shown"
else
    echo "⚠️  Warning: Could not find Info.plist at $INFO_PLIST"
fi

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