# Code Changes and Edits

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