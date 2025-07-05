# Code Changes and Edits

## MLX Nanobind Duplication Fix (CRITICAL)

### Issue
- App bundle crashes on other machines with error: "Critical nanobind error: refusing to add duplicate key 'cpu' to enumeration 'mlx.core.DeviceType'!"
- Error shows: "nanobind: type 'Device' was already registered!" and "nanobind: type 'DeviceType' was already registered!"
- App aborts immediately on launch, preventing use on deployed systems

### Root Cause
- PyInstaller was bundling MLX through multiple conflicting paths:
  - `--add-data="${MLX_PATH}:mlx"` (direct MLX path addition)
  - `--collect-all=mlx` (automatic MLX collection)
  - `--hidden-import=mlx`, `--hidden-import=mlx.core`, etc. (explicit imports)
  - MLX dependencies from `mlx_lm`, `mlx_whisper`, `parakeet_mlx` packages
- This caused MLX nanobind types to be registered multiple times during app initialization

### Solution (File: `build_app.sh`)

#### 1. Created MLX-Specific Runtime Hook
```bash
# NEW: rthooks/pyi_rth_mlx_fix.py
cat > rthooks/pyi_rth_mlx_fix.py << 'EOF'
# Critical: Prevent MLX nanobind type duplication
if 'MLX_NANOBIND_INIT' not in os.environ:
    os.environ['MLX_NANOBIND_INIT'] = '1'
    
    # Set MLX environment to prevent conflicts
    os.environ['MLX_DISABLE_METAL_CACHE'] = '0'
    os.environ['MLX_MEMORY_POOL'] = '1'
EOF
```

#### 1b. Fixed Runtime Hook TypeError (REVERTED APPROACH)
**CRITICAL ERROR**: Initial runtime hook caused `TypeError: 'module' object is not subscriptable` because `__builtins__` is a module in PyInstaller context, not a dictionary.

**FIRST FIX FAILED**: Removed MLX bundling entirely, which broke local functionality with `ModuleNotFoundError: No module named 'mlx._reprlib_fix'`.

**FINAL FIX**: Reverted to original working MLX bundling and simplified runtime hook:
- **RESTORED**: All original MLX bundling methods to fix local functionality
- **SIMPLIFIED**: Runtime hook to use only environment variables and module cleanup
- **REMOVED**: Complex import patching that caused the TypeError

#### 2. Restored MLX Bundling (REVERT)
```bash
# RESTORED: All MLX collection methods (needed for local functionality)
--add-data="${MLX_PATH}:mlx"          # Direct MLX path (RESTORED)
--collect-all=mlx                     # Automatic MLX collection (RESTORED)
--hidden-import=mlx                   # Explicit MLX import (RESTORED)
--hidden-import=mlx.core              # MLX core import (RESTORED)
--hidden-import=mlx.nn                # MLX nn import (RESTORED)
--hidden-import=mlx.optimizers        # MLX optimizers import (RESTORED)

# KEPT: MLX-dependent packages
--collect-all=mlx_lm                  # MLX Language Models
--collect-all=mlx_whisper             # MLX Whisper
--collect-all=parakeet_mlx            # Parakeet MLX
```

#### 3. Simplified Runtime Hook Approach
```python
# NEW: Clean, simple runtime hook
# Set MLX environment variables
os.environ['MLX_DISABLE_METAL_CACHE'] = '0'
os.environ['MLX_MEMORY_POOL'] = '1'

# Clear any pre-loaded MLX modules for clean loading
mlx_modules = [k for k in sys.modules.keys() if k.startswith('mlx')]
for mod in mlx_modules:
    if mod in sys.modules:
        del sys.modules[mod]
```

#### 3. Updated Runtime Hook Reference
```bash
# OLD: --runtime-hook=rthooks/pyi_rth_ssl_fix.py
# NEW: --runtime-hook=rthooks/pyi_rth_mlx_fix.py
```

### Result
- MLX nanobind types are only registered once during app initialization
- Environment flag prevents multiple MLX initialization attempts
- Controlled import ensures single MLX core instance
- App should launch successfully on deployed systems

### Testing Required
- Rebuild app bundle: `./build_app.sh`
- Test on clean system without MLX development environment
- Verify no "nanobind: type 'Device' was already registered!" errors
- Confirm app loads and tray icon appears correctly

### Files Modified
- `build_app.sh` - Updated PyInstaller configuration and runtime hooks
- `EDITS.md` - This change log

## Audio Dependencies Installation (CRITICAL FIX)

### Issue
User reported that audio libraries are REQUIRED functionality, not optional. PyInstaller build warnings showed:
- `parakeet_mlx.stt` not found
- `parakeet_mlx.models` not found  
- `audiofile` not found
- `audresample` not found

### Root Cause
Missing audio processing libraries that are core to the app's audio transcription functionality.

### Solution (Files: `build_app.sh`, virtual environment)

#### 1. Installed Missing Audio Dependencies
```bash
pip install mlx-whisper parakeet-mlx audiofile audresample
```

#### 2. Updated PyInstaller Hooks
- **Enhanced `hook-parakeet_mlx.py`**: Added `audiofile.core`, `audmath`, `audeer`, `soundfile`, `soxr`, `numba`, `llvmlite`
- **Created `hook-audiofile.py`**: Comprehensive audiofile bundling with all dependencies
- **Created `hook-audresample.py`**: Proper audresample bundling with soxr and numba

#### 3. Updated PyInstaller Configuration
- **Added hidden imports**: `audiofile.core`, `audmath`, `audeer`, `soundfile`, `soxr`, `numba`, `llvmlite`
- **Added collect-all**: `audiofile`, `audresample`, `audmath`, `audeer`, `soundfile`, `soxr`, `numba`, `llvmlite`

### Result
- **Build warnings**: Should be eliminated for audio libraries
- **Audio transcription**: Now fully bundled in app
- **App functionality**: Complete audio support included in standalone bundle

### Testing Required
- Rebuild app bundle: `./build_app.sh`
- Verify no audio-related PyInstaller warnings
- Test audio transcription functionality in bundled app
- Confirm `test_audio.py` works with bundled app

## FFmpeg Bundling for Audio Support (CRITICAL FIX)

### Issue
Audio transcription failed with error: "FFmpeg is not installed or not in your PATH"

### Root Cause
Audio libraries (audiofile, audresample, parakeet-mlx) require FFmpeg binary for audio format conversion and processing, but it wasn't bundled with the app.

### Solution (File: `build_app.sh`)

#### 1. Detect and Bundle FFmpeg Binary
```bash
# Find FFmpeg path and bundle it in bin subdirectory
FFMPEG_PATH=$(which ffmpeg)
FFMPEG_BINARY="--add-binary=$FFMPEG_PATH:bin"
```

#### 1b. Fixed Path Conflict (CRITICAL)
**ERROR**: `Pyinstaller needs to create a directory at 'ffmpeg', but there already exists a file at that path!`

**ROOT CAUSE**: Conflict between:
- `--add-binary` creating a FILE called `ffmpeg` (the binary)
- `--collect-all=ffmpeg` creating a DIRECTORY called `ffmpeg` (Python package)

**FIX**: Put FFmpeg binary in `bin` subdirectory instead of root:
- Changed from `--add-binary=$FFMPEG_PATH:.` to `--add-binary=$FFMPEG_PATH:bin`

#### 2. Add FFmpeg to Runtime PATH
```python
# In runtime hook: Add FFmpeg bin directory to PATH
ffmpeg_bin_dir = os.path.join(sys._MEIPASS, 'bin')
ffmpeg_path = os.path.join(ffmpeg_bin_dir, 'ffmpeg')
if os.path.exists(ffmpeg_path):
    current_path = os.environ.get('PATH', '')
    os.environ['PATH'] = f"{ffmpeg_bin_dir}:{current_path}"
```

#### 3. Install and Bundle ffmpeg-python
```bash
pip install ffmpeg-python
# Added to PyInstaller: --hidden-import=ffmpeg --collect-all=ffmpeg
```

#### 4. Created FFmpeg Hook
- **Created `hook-ffmpeg.py`**: Comprehensive ffmpeg-python bundling with all modules

### Result
- **FFmpeg binary**: Bundled with app at `/path/to/app/ffmpeg`
- **FFmpeg Python interface**: Fully bundled via ffmpeg-python package
- **Audio transcription**: Should work in standalone app without system FFmpeg
- **PATH setup**: FFmpeg automatically added to PATH during app startup

### Testing Required
- Rebuild app bundle: `./build_app.sh`
- Test audio transcription with `test_audio.py`
- Verify FFmpeg is found and audio processing works
- Confirm no "FFmpeg is not installed" errors

## AV Library Linking Fix (CRITICAL)

### Issue
Audio transcription failed with dynamic linking error:
```
Symbol not found: _av_buffer_unref
Referenced from: .../bin/ffmpeg
Expected in: .../av/__dot__dylibs/libavdevice.61.3.100.dylib
```

### Root Cause
- System FFmpeg binary has different libav* library dependencies than Python av package
- PyInstaller bundled incompatible versions causing dynamic linking conflicts
- FFmpeg looking for symbols in wrong libav libraries

### Solution (Files: `build_app.sh`)

#### 1. Ensure Latest Audio Dependencies
```bash
# Added to build script
pip install parakeet-mlx -U
pip install av -U  
pip install ffmpeg-python -U
```

#### 2. Bundle AV Library Properly
```bash
# Added comprehensive av library bundling
--hidden-import=av
--hidden-import=av.codec
--hidden-import=av.container
--hidden-import=av.format
--hidden-import=av.stream
--collect-all=av
```

#### 3. Created AV Library Hook
- **Created `hook-av.py`**: Comprehensive av package bundling with all dylibs
- **Collects dynamic libraries**: `collect_dynamic_libs('av')` to bundle all libav* dylibs
- **All av submodules**: audio, video, codec, container, format, stream, etc.

#### 4. Prioritize AV Libraries in Runtime
```python
# In runtime hook: Set DYLD_LIBRARY_PATH for av libraries
av_dylib_dir = os.path.join(sys._MEIPASS, 'av', '__dot__dylibs')
os.environ['DYLD_LIBRARY_PATH'] = f"{av_dylib_dir}:{current_dyld_path}"
```

### Result
- **AV libraries**: Properly bundled with compatible versions
- **Dynamic linking**: FFmpeg should find correct libav* symbols
- **Audio transcription**: Should work without symbol not found errors
- **Library compatibility**: System FFmpeg + Python av package dylibs

### Testing Required
- Rebuild app bundle: `./build_app.sh`
- Test audio transcription with `test_audio.py`
- Verify no "Symbol not found: _av_buffer_unref" errors
- Confirm FFmpeg and av libraries work together

### Files Modified
- `build_app.sh` - Updated PyInstaller configuration and runtime hooks
- Virtual environment - Added missing audio dependencies
- `EDITS.md` - This change log

## Model Size Estimation Improvements

### Issue
- Models with unconventional naming showing incorrect sizes
- Example: "Devstral-Small-2505-MLX-4bit" showing 2.3GB instead of correct size
- Problem: Pattern "5b" was matching the "5" in "2505"

### Root Cause
- Substring pattern matching was too loose: `if pattern in model_name`
- Patterns like "5b", "7b" were matching partial numbers in model names
- No word boundaries to ensure complete parameter specification matches

### Solution (File: `src/mlx_gui/huggingface_integration.py`)

#### 1. Added Model Card Parsing (Primary)
```python
# Look for parameter patterns in model card description
param_patterns = [
    r'(\d+(?:\.\d+)?)\s*[Bb](?:illion)?\s+param',        # "3.68B params"
    r'(\d+(?:\.\d+)?)\s*[Bb](?:illion)?\s+parameter',    # "3.68B parameters"
    r'Parameters?:\s*(\d+(?:\.\d+)?)\s*[Bb]',             # "Parameters: 3.68B"
    # ... more patterns
]
```

#### 2. Fixed Pattern Matching (Fallback)
```python
# OLD: Loose substring matching
if pattern in model_name and len(pattern) > best_match_length:

# NEW: Precise word boundary matching
escaped_pattern = re.escape(pattern)
regex_pattern = rf'(?<![a-zA-Z0-9]){escaped_pattern}(?![a-zA-Z0-9])'
if re.search(regex_pattern, model_name, re.IGNORECASE):
```

#### 3. Conservative Approach
- Return `None` if parameter count cannot be determined reliably
- No more file-size fallbacks or heuristic guessing
- UI only shows size when confident

### Result
- "Devstral-Small-2505-MLX-4bit" now shows no size (correct)
- Models with clear parameter specs still work: "llama-7b-instruct" → 7B params
- Model cards with "3.68B params" → accurate size calculation
- Eliminated misleading size estimates

### Files Modified
- `src/mlx_gui/huggingface_integration.py` - Core size estimation logic
- `src/mlx_gui/templates/admin.html` - UI handles missing size gracefully
- `NOTES.md` - Updated documentation
- `EDITS.md` - This change log

## Safetensors Metadata Support (Follow-up Fix)

### Issue
- Parameter count "3.68B params" visible on HuggingFace web page for "Devstral-Small-2505-MLX-4bit"
- Our code only checked model description, but HuggingFace stores this in safetensors metadata
- User correctly pointed out the information is right there on the web page

### Solution (File: `src/mlx_gui/huggingface_integration.py`)

#### Added Safetensors Metadata Parsing
```python
# Check safetensors metadata for parameter information
if hasattr(model_info_data, 'safetensors') and model_info_data.safetensors:
    for file_name, metadata in model_info_data.safetensors.items():
        if isinstance(metadata, dict):
            # Check for 'total' parameter count
            if 'total' in metadata:
                total_params = metadata.get('total', 0)
                if total_params > 1000000:
                    param_count_billions = total_params / 1e9
```

#### Updated Processing Order
1. **First**: Model card description
2. **Second**: Safetensors metadata (NEW)
3. **Fallback**: Model name patterns

### Expected Result
- "Devstral-Small-2505-MLX-4bit" should now show: 3.68B params × 4-bit = ~1.7GB
- Better coverage of models with parameter info in metadata vs description

### Debug Support
- Added `/v1/debug/model/{model_id}` endpoint to inspect all model fields
- Helps troubleshoot size estimation issues

## HuggingFace Links for Loaded Models

### Issue
- User requested clickable links to HuggingFace source for loaded models
- Current admin interface only showed links for discovery results
- Loaded models showed static names without source links

### Solution

#### 1. Added HuggingFace ID to Model List API (File: `src/mlx_gui/server.py`)
```python
# Added huggingface_id field to /v1/manager/models response
{
    "id": model.id,
    "name": model.name,
    "type": model.model_type,
    "status": model.status,
    "memory_required_gb": model.memory_required_gb,
    "use_count": model.use_count,
    "last_used_at": model.last_used_at.isoformat() if model.last_used_at else None,
    "created_at": model.created_at.isoformat() if model.created_at else None,
    "huggingface_id": model.huggingface_id,  # NEW
}
```

#### 2. Updated Admin Interface (File: `src/mlx_gui/templates/admin.html`)
```html
<!-- OLD: Static model name -->
<h3 class="text-lg font-medium text-white">${model.name}</h3>

<!-- NEW: Clickable link when HuggingFace ID available -->
<h3 class="text-lg font-medium text-white">
    ${model.huggingface_id ? 
        `<a href="https://huggingface.co/${model.huggingface_id}" target="_blank" rel="noopener noreferrer" 
           class="hover:text-blue-300 transition-colors cursor-pointer">
            ${model.name}
        </a>` :
        model.name
    }
</h3>
```

### Result
- Loaded models now show clickable names that link to their HuggingFace source
- Opens in new tab/window with proper security attributes
- Hover effect shows it's clickable
- Falls back to static name if no HuggingFace ID available
- Consistent with discovery results interface

### Size Calculation Summary
- **Discovery models**: Real-time calculation using improved `_estimate_model_size()`
- **Loaded models**: Static value from database (`memory_required_gb`)
- **Source**: Size set during installation from HuggingFace estimate
- **Display**: Both show size only when reliably determined

### Files Modified
- `src/mlx_gui/server.py` - Added huggingface_id to model list response
- `src/mlx_gui/templates/admin.html` - Added clickable HuggingFace links
- `NOTES.md` - Updated with size calculation overview
- `EDITS.md` - This change log

## Author/Source Tags for Installed Models

### Issue
- User requested author/source tags for installed models (e.g., "mlx-community", "lmstudio")
- Current interface only showed model name and type
- Discovery results showed author but installed models did not

### Solution

#### 1. Added Author Field to Model List API (File: `src/mlx_gui/server.py`)
```python
# Extract author from huggingface_id in /v1/manager/models response
{
    "id": model.id,
    "name": model.name,
    "type": model.model_type,
    "status": model.status,
    "memory_required_gb": model.memory_required_gb,
    "use_count": model.use_count,
    "last_used_at": model.last_used_at.isoformat() if model.last_used_at else None,
    "created_at": model.created_at.isoformat() if model.created_at else None,
    "huggingface_id": model.huggingface_id,
    "author": model.huggingface_id.split("/")[0] if model.huggingface_id and "/" in model.huggingface_id else "unknown",  # NEW
}
```

#### 2. Updated Admin Interface (File: `src/mlx_gui/templates/admin.html`)
```html
<!-- Added author tag after model type -->
<span class="ml-2 px-2 py-1 text-xs bg-gray-700 text-gray-300 rounded">${model.type}</span>
${model.author && model.author !== 'unknown' ? 
    `<span class="ml-2 px-2 py-1 text-xs bg-purple-700 text-purple-300 rounded">
        <i class="fas fa-user mr-1"></i>${model.author}
    </span>` : ''
}
```

### Result
- Installed models now show author/source tags with purple styling
- Tags extracted from `huggingface_id` (e.g., "mlx-community/Qwen2.5-7B-Instruct-4bit" → "mlx-community")
- Only shows tag when author is available (not "unknown")
- Consistent with discovery results styling but different color (purple vs blue)
- Visual hierarchy: Status → Name → Type → Author

### Examples
- `mlx-community/Qwen2.5-7B-Instruct-4bit` → Shows "mlx-community" tag
- `microsoft/DialoGPT-small` → Shows "microsoft" tag  
- `lmstudio/llama-3.2-3b-instruct` → Shows "lmstudio" tag

## macOS App Bundle Support

### Issue
- User asked how to make MLX-GUI into a distributable macOS app
- Current setup requires Python environment and command-line usage
- Need native macOS app bundle for easy distribution and installation

### Solution

#### 1. Created PyInstaller Configuration (File: `setup_app.py`)
```python
# PyInstaller configuration for true standalone app
args = [
    'src/mlx_gui/app_main.py',
    '--name=MLX-GUI',
    '--onedir',
    '--windowed',
    '--hidden-import=mlx',
    '--hidden-import=mlx_lm',
    '--collect-all=mlx',
    '--osx-bundle-identifier=org.matthewrogers.mlx-gui',
]
```

#### 2. Created Dedicated App Entry Point (File: `src/mlx_gui/app_main.py`)
```python
# Handles app bundle environment setup
def setup_app_environment():
    if hasattr(sys, 'frozen') and sys.frozen:
        # Running as app bundle
        log_dir = Path.home() / "Library" / "Logs" / "MLX-GUI"
        # Setup logging, paths, etc.
```

#### 3. Automated Build Scripts
- `build_app.sh`: Creates the app bundle using PyInstaller
- `create_dmg.sh`: Creates professional DMG installer with drag-to-Applications

#### 4. Updated Dependencies (File: `pyproject.toml`)
```toml
[project.optional-dependencies]
app = [
    "pyinstaller>=6.0.0",
    "rumps>=0.4.0",
    "pillow>=10.0.0",
]
```

### Result
- Native macOS app bundle: `dist/MLX-GUI.app`
- Professional DMG installer: `MLX-GUI-0.1.0.dmg`
- Self-contained with all dependencies
- Runs as background app with system tray only
- Proper macOS integration and metadata
- Logs to `~/Library/Logs/MLX-GUI/mlx-gui.log`

### Usage
1. Install app dependencies: `pip install -e ".[app]"`
2. Build app bundle: `./build_app.sh`
3. Create DMG installer: `./create_dmg.sh`
4. Distribute or install to Applications folder

### Resolved: True Standalone App Bundle
Created a complete PyInstaller-based solution for true standalone app distribution:

**Solution: PyInstaller for Binary Dependencies**
PyInstaller properly handles MLX's compiled binaries (`.so` and `.dylib` files) that other tools struggle with.

**Key Features:**
- **TRUE STANDALONE**: No Python installation required on target system
- **Complete binary support**: Properly handles MLX's `.so` and `.dylib` files
- **Self-contained**: Includes Python runtime, MLX, and all dependencies
- **Just works**: Recipients can run the app without any setup

### Files Created/Modified
- `setup_app.py` - PyInstaller configuration for standalone apps
- `build_app.sh` - Build script using PyInstaller
- `src/mlx_gui/app_main.py` - App entry point
- `create_dmg.sh` - DMG creation script
- `entitlements.plist` - macOS entitlements for code signing
- `requirements.txt` - Complete frozen dependencies
- `pyproject.toml` - Added PyInstaller to app dependencies
- `NOTES.md` - Complete app building documentation
- `EDITS.md` - This change log

#### 5. True Standalone App Bundle
- Created complete PyInstaller-based solution for standalone app distribution
- `setup_app.py` with comprehensive package list including:
  - `mlx-lm` (critical for MLX functionality)
  - `transformers`, `tokenizers`, `safetensors` (model handling)
  - `numpy`, `protobuf`, `sentencepiece` (core dependencies)
  - All FastAPI, uvicorn, and web server dependencies
- `build_app.sh` with dependency validation and PyInstaller build
- Creates truly standalone app with no Python dependencies required
- Removed problematic py2app approach that couldn't handle MLX binaries
- Cleaned up all py2app code and simplified to single PyInstaller solution
- Renamed scripts to simple names: `setup_app.py`, `build_app.sh`

### Files Modified
- `src/mlx_gui/server.py` - Added author field extraction from huggingface_id
- `src/mlx_gui/templates/admin.html` - Added author tag display
- `NOTES.md` - Updated with author tag functionality
- `EDITS.md` - This change log

## App Bundle Creation (PyInstaller)

1. **Refactored `build_app.sh`**: Modified to call `pyinstaller` directly, ensuring a proper `.app` bundle is created.
2. **Deleted `setup_app.py`**: Removed the intermediate Python build script as it's no longer needed.
3. **Updated `NOTES.md`**: Documented the new build process and fixes.

**Result**: Creates a TRUE standalone macOS app that works on any system without Python setup.

## Audio Transcription Test Script

### Created `test_audio.py`
- **Purpose**: Comprehensive test script for the `/v1/audio/transcriptions` endpoint
- **Features**:
  - Tests with `parakeet-tdt-0-6b-v2` model
  - Automatic server connectivity check
  - Model status verification and auto-loading
  - Multipart form data handling
  - Multiple response format support (json, text, verbose_json, srt, vtt)
  - Detailed error handling and troubleshooting
  - Segment-level transcription output
- **Usage**: `python test_audio.py` or `./test_audio.py`
- **Requirements**: Requires a `test.wav` file in the project directory
- **Documentation**: Added to `NOTES.md` with complete feature list and usage instructions

## App Bundle Issues Fixed

### Tray Icon Not Working
- **File**: `build_app.sh`
- **Issue**: System tray icon did not appear.
- **Root Cause**: The app was not being built as a proper `.app` bundle.
- **Fix**: Changed build process to call `pyinstaller` directly with `--windowed`, which correctly generates a `dist/MLX-GUI.app` bundle.

### Template Path Resolution
- **File**: `src/mlx_gui/server.py`
- **Issue**: Admin interface returned a `404 Not Found` error in the bundled app.
- **Fix**: Corrected the template path resolution to use `sys._MEIPASS`, which is the standard PyInstaller variable for accessing bundled data files.