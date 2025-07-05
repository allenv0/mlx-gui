# MLX-GUI Development Notes

## Current State

### Model Size Calculation Overview

**For Discovery Models** (Discover tab):
- Size calculated in real-time using `_estimate_model_size()` in `huggingface_integration.py`
- Uses model card descriptions, safetensors metadata, and name patterns
- Returns `None` if parameter count cannot be determined reliably

**For Loaded Models** (Models tab):
- Size comes from `memory_required_gb` field in database
- Set once during model installation from HuggingFace size estimate
- Static value - doesn't change after installation
- Displayed as clickable links to HuggingFace source when `huggingface_id` is available
- Shows author/source tags extracted from `huggingface_id` (e.g., "mlx-community", "lmstudio")

### Model Size Estimation (Updated)
- **Approach**: Conservative size estimation - only show size when we can confidently determine it
- **Implementation**: `_estimate_model_size()` in `huggingface_integration.py`
- **Strategy**: 
  - **First**: Extract parameter count from model card description using regex patterns
    - Patterns like "3.68B params", "3.68B parameters", "Parameters: 3.68B", etc.
    - Much more accurate than name-based guessing
  - **Second**: Check safetensors metadata for parameter count (where HuggingFace stores "3.68B params")
    - Looks for 'total' parameter count in safetensors metadata
    - Searches metadata strings for parameter patterns
  - **Fallback**: Extract parameter count from model name using **precise** pattern matching
    - Uses word boundaries to prevent false matches (e.g., "5b" matching in "2505")
    - Only matches complete parameter specifications like "7b", "13b", etc.
  - Calculate memory based on parameter count and quantization level
  - Return `None` if parameter count cannot be determined reliably
  - Admin UI only displays size when available and > 0
- **Patterns**: Extended parameter patterns to catch common model naming conventions
- **No Guessing**: Removed file-size fallbacks and heuristic guessing to prevent misleading estimates
- **Model Card Priority**: Model card content is checked first as it's more reliable than model names
- **Fixed False Matches**: Pattern matching now uses word boundaries to prevent matching partial numbers
- **Safetensors Support**: Added safetensors metadata parsing to access parameter counts shown on HuggingFace web pages

### Functions and Definitions

#### HuggingFace Integration (`huggingface_integration.py`)
- `_estimate_model_size(model)`: Conservative size estimation, returns None if uncertain
- `_extract_model_info(model)`: Extracts model metadata including size
- `search_mlx_models()`: Search for MLX-compatible models
- `get_popular_mlx_models()`: Get popular models by downloads
- `get_model_details(model_id)`: Get detailed model information

#### Admin Interface (`admin.html`)
- `renderDiscoveryResults()`: Updated to handle missing size data gracefully
- `loadPopularModels()`: Load popular models from HuggingFace
- `searchModels()`: Search models with query
- `installModel()`: Install model from HuggingFace

#### Server Endpoints (`server.py`)
- `/v1/discover/models`: Search models with size filtering
- `/v1/discover/popular`: Get popular models
- `/v1/discover/compatible`: Get system-compatible models
- `/v1/models/install`: Install model from HuggingFace

### Model Size Display Logic
- Only display size when `size_gb` is available and > 0
- No placeholder or "unknown" text shown
- Clean interface that doesn't mislead users about model requirements

### Recent Changes
- Simplified size estimation to prevent misleading data
- Removed complex heuristics and file-based fallbacks
- Updated UI to gracefully handle missing size information
- Focused on accuracy over completeness for size data
- **Fixed critical bug**: Pattern matching now uses word boundaries to prevent false matches
  - Issue: "5b" was matching in "2505" giving wrong size estimates
  - Fix: Regex with word boundaries ensures only complete parameter specs match
- Added model card parsing as primary source for parameter counts
- **Added safetensors metadata support**: Now extracts parameter counts from safetensors metadata
  - Addresses issue where "3.68B params" visible on HuggingFace page wasn't being detected
  - Checks safetensors metadata where HuggingFace stores model specifications
  - Should correctly detect "Devstral-Small-2505-MLX-4bit" as 3.68B params → ~1.7GB with 4-bit

## Design Principles
- Avoid misleading users with inaccurate size estimates
- Show only reliable, verified information
- Clean UI that handles missing data gracefully
- Conservative approach to prevent system compatibility issues

## macOS App Bundle

### Building the App
MLX-GUI can be packaged as a native macOS app bundle for easy distribution using PyInstaller.

**Files:**
- `build_app.sh`: Build script that creates the app bundle directly with PyInstaller.
- `create_dmg.sh`: Script to create a DMG installer.
- `src/mlx_gui/app_main.py`: Dedicated entry point for the macOS app.
- `entitlements.plist`: macOS entitlements for code signing (optional).

**Build Process:**
1. Install dependencies: `pip install -e ".[app]"` or `pip install -r requirements.txt`
2. Build TRUE standalone app: `./build_app.sh`
3. Create DMG installer: `./create_dmg.sh`
- `requirements.txt`: Complete frozen dependencies from pip freeze.
- Includes critical packages: `mlx-lm`, `mlx`, `transformers`, `huggingface_hub`, etc.
- Build script validates all critical dependencies are installed.

**Output:**
- `dist/MLX-GUI.app`: Native macOS app bundle.
- `MLX-GUI-0.1.0.dmg`: Distributable DMG installer.
- Runs as background app (system tray only, no dock icon).
- **TRUE STANDALONE** - No Python installation required!
- Includes complete Python runtime and all dependencies.
- Self-contained MLX binaries and libraries.
- Proper macOS integration with bundle ID and metadata.
- Logs to `~/Library/Logs/MLX-GUI/mlx-gui.log`.
- Can be installed by dragging to Applications folder.
- **Works on any macOS system - NO SETUP REQUIRED!**

### Distribution Options
1. **Direct app sharing**: Share the `MLX-GUI.app` bundle directly - **JUST WORKS!** ✨
2. **DMG installer**: Professional installer with drag-to-Applications.
3. **Signed & notarized**: For Mac App Store or public distribution (requires Apple Developer account).

**🎉 True Standalone Success:** Recipients need NO Python setup, NO virtual environments, NO dependencies. Just download and run!

## App Bundle Issues Fixed

### Tray Icon Not Working
- **Issue**: System tray icon did not appear in the menu bar.
- **Root Cause**: The app was not being built as a proper `.app` bundle. Running the raw executable prevents macOS from correctly handling GUI elements like the tray.
- **Fix**: Refactored `build_app.sh` to call `pyinstaller` directly, ensuring it creates a standard `.app` bundle. Deleted the intermediate `setup_app.py` script. The build script now correctly verifies the creation of `dist/MLX-GUI.app`.
- **Result**: The app now builds as a proper macOS application, allowing the tray icon to work as expected.

### Template Path Resolution
- **Issue**: Admin template not found (`404 Error`) in the bundled app.
- **Fix**: Modified `server.py` to use `sys._MEIPASS`, the standard PyInstaller method for finding bundled data files.
- **Result**: Admin interface now loads correctly in the standalone app.