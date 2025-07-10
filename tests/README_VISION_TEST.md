# MLX-GUI Multimodal Vision Test

This test verifies that MLX-GUI can properly handle multimodal vision models with image inputs.

## Test Overview

The test script (`test_multimodal_vision.py`) performs the following:

1. **Server Health Check** - Verifies MLX-GUI server is running
2. **Model Installation** - Installs `mlx-community/gemma-3n-E4B-it-bf16` 
3. **Model Loading** - Loads the vision model into memory
4. **Image Processing** - Encodes `icon.png` to base64
5. **Multimodal Request** - Sends image + text prompt via `/v1/chat/completions`
6. **Response Validation** - Checks if model can see the "M" in the image

## Prerequisites

1. **MLX-GUI Server Running:**
   ```bash
   mlx-gui start --port 8000
   ```

2. **MLX-VLM Installed:**
   ```bash
   pip install mlx-vlm>=0.1.0
   # Or: pip install mlx-gui[vision]
   ```

3. **Sufficient Memory:**
   - Gemma-3n E4B model requires ~8-12GB RAM
   - Ensure you have enough available memory

## Running the Test

### Option 1: Use the test runner (recommended)
```bash
cd /path/to/mlx-gui
./tests/run_vision_test.sh
```

### Option 2: Run directly with Python
```bash
cd /path/to/mlx-gui
python tests/test_multimodal_vision.py
```

## Expected Output

If successful, you should see output like:
```
✅ Server is running and healthy
📥 Installing model mlx-community/gemma-3n-E4B-it-bf16...
✅ Model installed: Model 'gemma-3n-e4b-it' installed successfully
🔄 Loading model gemma-3n-e4b-it...
✅ Model loaded: Model 'gemma-3n-e4b-it' loaded successfully
✅ Model is healthy and ready
👁️  Testing multimodal vision with /path/to/icon.png...
📸 Encoded image (icon.png) to base64 (XXXX chars)
🤖 Sending multimodal request to model...
✅ Multimodal response received!
   Duration: 12.34s
   Tokens: 89 prompt + 156 completion = 245 total
   Response length: 423 characters

📝 Model Response:
==================================================
I can see a large white letter "M" on a dark gray background. The letter appears to be in a clean, modern font style and takes up most of the image space. Below the main "M" logo, there appears to be text that reads "MLX" in smaller letters.
==================================================

🎯 SUCCESS: Model correctly identified the 'M' in the image!
🎉 Multimodal vision test completed successfully!
```

## Test Image

The test uses `icon.png` from the project root, which contains:
- A white letter "M" on dark background  
- "MLX" text below
- Perfect test case for vision model verification

## Troubleshooting

### Server Not Running
```
❌ Cannot connect to server: Connection refused
   Make sure MLX-GUI server is running: mlx-gui start
```
**Solution:** Start the MLX-GUI server first

### Model Installation Fails
```
❌ Model installation failed: 400
   Response: {"detail": "Insufficient memory..."}
```
**Solution:** Free up system memory or use a smaller model

### MLX-VLM Not Installed
```
❌ mlx-vlm not installed - Vision/multimodal models not supported
```
**Solution:** Install MLX-VLM: `pip install mlx-vlm`

### Model Doesn't See Image
```
⚠️  Model response doesn't explicitly mention 'M'
```
**Check:** 
- Model logs for vision processing
- Image encoding (should be ~30KB+ base64)
- Model type detection in server logs

## Architecture Notes

The test validates the complete multimodal pipeline:
- **MLX-VLM Integration** - Vision model loading and inference  
- **Image Processing** - Base64 encoding and temporary file handling
- **OpenAI Compatibility** - Chat completions with multimodal content
- **Queue Management** - Transparent queuing for vision requests
- **Error Handling** - Graceful fallbacks and proper cleanup

This ensures the Gemma-3 vision model can properly "see" images from chat clients, resolving the original issue.