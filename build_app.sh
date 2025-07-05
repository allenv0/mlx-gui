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
CRITICAL_DEPS=("mlx-lm" "mlx" "rumps" "fastapi" "uvicorn" "transformers" "huggingface-hub" "mlx-whisper" "parakeet-mlx")
MISSING_DEPS=""

for dep in "${CRITICAL_DEPS[@]}"; do
    if ! pip show "$dep" > /dev/null 2>&1; then
        MISSING_DEPS="$MISSING_DEPS $dep"
    fi
done

if [ -n "$MISSING_DEPS" ]; then
    echo "❌ Missing critical dependencies:$MISSING_DEPS"
    echo "💡 Install with: pip install -e \".[app,audio]\""
    echo "💡 Or from requirements: pip install -r requirements.txt"
    echo "💡 For audio support: pip install mlx-whisper parakeet-mlx"
    exit 1
fi

echo "✅ All critical dependencies found"

# Ensure latest audio dependencies
echo "📦 Ensuring latest audio dependencies..."
pip install parakeet-mlx -U
pip install av -U
pip install ffmpeg-binaries -U

# Clean previous builds
echo "🧹 Cleaning previous builds..."
pkill -f MLX-GUI || true
sleep 2
rm -rf build/ dist/ MLX-GUI.spec app_icon.icns 2>/dev/null || true

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

# Set environment variables to prevent model downloads during build
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_DISABLE_CUDA_MALLOC=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Create PyInstaller hooks directory if it doesn't exist
mkdir -p hooks

# Create custom hook for parakeet-mlx
cat > hooks/hook-parakeet_mlx.py << 'EOF'
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = collect_all('parakeet_mlx')

# Additional hidden imports for parakeet-mlx and its dependencies
hiddenimports += [
    'parakeet_mlx.stt',
    'parakeet_mlx.models', 
    'parakeet_mlx.utils',
    'parakeet_mlx.alignment',
    'parakeet_mlx.attention',
    'parakeet_mlx.audio',
    'parakeet_mlx.cache',
    'parakeet_mlx.conformer',
    'parakeet_mlx.ctc',
    'parakeet_mlx.rnnt',
    'parakeet_mlx.tokenizer',
    'dacite',
    'librosa',
    'librosa.core',
    'librosa.feature',
    'librosa.util',
    'typer',
    'audiofile',
    'audiofile.core',
    'audresample',
    'audmath',
    'audeer',
    'soundfile',
    'soxr',
    'numba',
    'llvmlite',
]
EOF

# Create custom hook for audiofile
cat > hooks/hook-audiofile.py << 'EOF'
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = collect_all('audiofile')

# Additional hidden imports for audiofile
hiddenimports += [
    'audiofile.core',
    'audmath',
    'audeer',
    'soundfile',
    'cffi',
    'pycparser',
]
EOF

# Create custom hook for audresample
cat > hooks/hook-audresample.py << 'EOF'
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = collect_all('audresample')

# Additional hidden imports for audresample
hiddenimports += [
    'soxr',
    'numba',
    'llvmlite',
]
EOF

# Note: Removed ffmpeg-python hook as we're using Python av package instead

# Create custom hook for av (PyAV)
cat > hooks/hook-av.py << 'EOF'
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

datas, binaries, hiddenimports = collect_all('av')

# Collect all av dynamic libraries (libav* dylibs)
av_dylibs = collect_dynamic_libs('av')
binaries.extend(av_dylibs)

# Additional hidden imports for av
hiddenimports += [
    'av',
    'av.audio',
    'av.codec',
    'av.container',
    'av.format',
    'av.stream',
    'av.video',
    'av.filter',
    'av.packet',
    'av.frame',
    'av.plane',
    'av.subtitles',
    'av.logging',
    'av.utils',
]
EOF

# Create custom hook for mlx-whisper
cat > hooks/hook-mlx_whisper.py << 'EOF'
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = collect_all('mlx_whisper')

# Additional hidden imports for mlx-whisper
hiddenimports += [
    'mlx_whisper.transcribe',
    'mlx_whisper.load_models',
    'mlx_whisper.audio',
]
EOF

# Create runtime hook for ffmpeg-binaries to ensure FFmpeg is in PATH
mkdir -p rthooks
cat > rthooks/pyi_rth_ffmpeg_binaries.py << 'EOF'
"""
PyInstaller runtime hook to initialize FFmpeg binaries from the
`ffmpeg-binaries` package (imported as `ffmpeg`).
This guarantees the bundled FFmpeg is added to PATH so libraries like
parakeet_mlx can find it when the app is launched from Finder.
"""
import os, shutil

def _setup_ffmpeg() -> None:
    try:
        import ffmpeg  # provided by ffmpeg-binaries
        ffmpeg.init()
        ffmpeg.add_to_path()
        bin_path = getattr(ffmpeg, 'FFMPEG_PATH', None)
        if bin_path and os.path.exists(bin_path):
            os.environ.setdefault('FFMPEG_BINARY', bin_path)
            found = shutil.which('ffmpeg')
            print(f"✅ FFmpeg initialized -> {found or bin_path}")
        else:
            print("⚠️  FFmpeg binary path not found after initialization")
    except Exception as exc:  # pragma: no cover
        print(f"⚠️  ffmpeg-binaries setup error: {exc}")

_setup_ffmpeg()
EOF

# Create runtime hook to fix SSL/crypto library conflicts and prevent MLX duplication
mkdir -p rthooks
cat > rthooks/pyi_rth_mlx_fix.py << 'EOF'
import os
import sys

# Fix SSL/crypto library conflicts by ensuring system libraries are prioritized
if hasattr(sys, '_MEIPASS'):
    # We're running in a PyInstaller bundle
    dylib_path = os.path.join(sys._MEIPASS, 'lib-dynload')
    if os.path.exists(dylib_path):
        # Remove problematic cv2 dylib paths from environment
        if 'DYLD_LIBRARY_PATH' in os.environ:
            paths = os.environ['DYLD_LIBRARY_PATH'].split(':')
            filtered_paths = [p for p in paths if 'cv2' not in p and 'opencv' not in p]
            os.environ['DYLD_LIBRARY_PATH'] = ':'.join(filtered_paths)
    
    # Setup av package's libav libraries for Python audio processing
    av_dylib_dir = os.path.join(sys._MEIPASS, 'av', '__dot__dylibs')
    if os.path.exists(av_dylib_dir):
        current_dyld_path = os.environ.get('DYLD_LIBRARY_PATH', '')
        os.environ['DYLD_LIBRARY_PATH'] = f"{av_dylib_dir}:{current_dyld_path}"
        print(f"✅ AV libraries available at: {av_dylib_dir}")
    else:
        print("⚠️  Warning: AV libraries not found in app bundle")
    
    # Try to prevent MLX nanobind conflicts with environment variables
    # Set these before any MLX imports happen
    os.environ['MLX_DISABLE_METAL_CACHE'] = '0'
    os.environ['MLX_MEMORY_POOL'] = '1'
    
    # Ensure clean MLX module loading
    mlx_modules = [k for k in sys.modules.keys() if k.startswith('mlx')]
    for mod in mlx_modules:
        if mod in sys.modules:
            del sys.modules[mod]
EOF

# Find MLX path for data files
# Using Python audio libraries only (av, librosa, soundfile)
# This avoids FFmpeg binary conflicts with Python av package
echo "📦 Using Python-only audio libraries (parakeet_mlx, av, librosa, soundfile)"
echo "   This avoids system FFmpeg vs Python av library conflicts"

# Check if we have a custom icon
ICON_PARAM=""
if [ -f "app_icon.icns" ]; then
    ICON_PARAM="--icon=app_icon.icns"
    echo "📱 Using custom app icon"
else
    echo "📱 Using default icon"
fi

# Read version from Python module
VERSION=$(python3 -c "from src.mlx_gui import __version__; print(__version__)")
echo "📝 Building version: $VERSION"

pyinstaller src/mlx_gui/app_main.py \
    --name="MLX-GUI" \
    --onedir \
    --windowed \
    --noconfirm \
    --clean \
    --additional-hooks-dir=hooks \
    --runtime-hook=rthooks/pyi_rth_ffmpeg_binaries.py \
    $ICON_PARAM \
    --exclude-module=cv2 \
    --exclude-module=opencv-python \
    --exclude-module=opencv-contrib-python \
    --exclude-module=torch.distributed \
    --exclude-module=torch.optim \
    --exclude-module=matplotlib \
    --hidden-import=scipy.sparse.csgraph._validation \
    --exclude-module=torch \
    --exclude-module=torchvision \
    --exclude-module=torchaudio \
    --exclude-module=tensorflow \
    --exclude-module=jax \
    --exclude-module=sklearn \
    --exclude-module=pandas \
    --exclude-module=IPython \
    --exclude-module=jupyter \
    --exclude-module=notebook \
    --exclude-module=bokeh \
    --exclude-module=plotly \
    --exclude-module=seaborn \
    --exclude-module=sympy \
    --hidden-import=mlx \
    --hidden-import=mlx_lm \
    --hidden-import=mlx.core \
    --hidden-import=mlx.nn \
    --hidden-import=mlx.optimizers \
    --hidden-import=mlx._reprlib_fix \
    --hidden-import=mlx.utils \
    --hidden-import=mlx_whisper \
    --hidden-import=mlx_whisper.transcribe \
    --hidden-import=parakeet_mlx \
    --hidden-import=dacite \
    --hidden-import=librosa \
    --hidden-import=typer \
    --hidden-import=audiofile \
    --hidden-import=audiofile.core \
    --hidden-import=audresample \
    --hidden-import=audmath \
    --hidden-import=audeer \
    --hidden-import=soundfile \
    --hidden-import=soxr \
    --hidden-import=numba \
    --hidden-import=llvmlite \
    --hidden-import=av \
    --hidden-import=av.codec \
    --hidden-import=av.container \
    --hidden-import=av.format \
    --hidden-import=av.stream \
    --hidden-import=ffmpeg \
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
    --add-data="src/mlx_gui/templates:mlx_gui/templates" \
    --collect-all=mlx \
    --collect-all=mlx_lm \
    --collect-all=mlx_whisper \
    --collect-all=parakeet_mlx \
    --collect-all=librosa \
    --collect-all=dacite \
    --collect-all=typer \
    --collect-all=audiofile \
    --collect-all=audresample \
    --collect-all=audmath \
    --collect-all=audeer \
    --collect-all=soundfile \
    --collect-all=soxr \
    --collect-all=numba \
    --collect-all=llvmlite \
    --collect-all=av \
    --collect-all=ffmpeg \
    --collect-all=transformers \
    --collect-all=tokenizers \
    --collect-all=safetensors \
    --collect-all=scipy.sparse.csgraph \
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
    
    # Add version information to Info.plist
    /usr/libexec/PlistBuddy -c "Add :CFBundleShortVersionString string $VERSION" "$INFO_PLIST" 2>/dev/null || \
    /usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $VERSION" "$INFO_PLIST"
    
    /usr/libexec/PlistBuddy -c "Add :CFBundleVersion string $VERSION" "$INFO_PLIST" 2>/dev/null || \
    /usr/libexec/PlistBuddy -c "Set :CFBundleVersion $VERSION" "$INFO_PLIST"
    
    echo "✅ App converted to menu bar app (no dock icon)"
    echo "   - App will only appear in the menu bar"
    echo "   - No dock icon will be shown"
    echo "   - Version set to: $VERSION"
else
    echo "⚠️  Warning: Could not find Info.plist at $INFO_PLIST"
fi

# Clean up temporary hook files
echo "🧹 Cleaning up temporary hook files..."
rm -rf hooks/ rthooks/

# Check if build was successful
if [ -d "dist/MLX-GUI.app" ]; then
    echo "✅ App bundle built successfully!"
    echo "📍 Location: dist/MLX-GUI.app"
    
    # Code signing section
    echo ""
    echo "🔐 Code Signing..."
    
    # Check if we have a Developer ID Application certificate
    CERT_NAME=$(security find-identity -v -p codesigning | grep "Developer ID Application" | head -1 | sed 's/.*"\(.*\)".*/\1/')
    
    if [ -n "$CERT_NAME" ]; then
        echo "📝 Found certificate: $CERT_NAME"
        echo "🔏 Signing app bundle..."
        
        # Sign all executables and libraries first (deep signing)
        codesign --force --deep --sign "$CERT_NAME" --options runtime --entitlements /dev/null "dist/MLX-GUI.app"
        
        # Verify the signature
        if codesign --verify --verbose "dist/MLX-GUI.app" 2>/dev/null; then
            echo "✅ App successfully signed!"
            echo "🛡️  This will eliminate macOS security warnings"
            
            # Show signature info
            echo ""
            echo "📜 Signature Info:"
            codesign -dv --verbose=4 "dist/MLX-GUI.app" 2>&1 | grep -E "(Identifier|TeamIdentifier|Authority)"
        else
            echo "⚠️  Warning: Code signing verification failed"
            echo "   The app was built but may show security warnings"
        fi
    else
        echo "⚠️  No Developer ID Application certificate found"
        echo "   App will show security warnings when downloaded"
        echo "   To fix this:"
        echo "   1. Get an Apple Developer account ($99/year)"
        echo "   2. Create a Developer ID Application certificate"
        echo "   3. Install it in Keychain Access"
        echo "   4. Re-run this build script"
    fi
    
    echo ""
    echo "🎉 You can now:"
    echo "   1. Run: open dist/MLX-GUI.app"
    echo "   2. Copy to /Applications: cp -R dist/MLX-GUI.app /Applications/"
    echo "   3. Create a DMG installer"
    echo ""
    echo "📋 App Info:"
    echo "   - Size: $(du -sh dist/MLX-GUI.app | cut -f1)"
    echo "   - Type: TRUE STANDALONE (no Python required!)"
    echo "   - Includes: All Python runtime, MLX binaries, audio support, and dependencies"
    if [ -n "$CERT_NAME" ]; then
        echo "   - Code Signed: ✅ (no security warnings)"
    else
        echo "   - Code Signed: ❌ (will show security warnings)"
    fi
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
echo "   • Audio support included: Whisper and Parakeet models work out of the box" 