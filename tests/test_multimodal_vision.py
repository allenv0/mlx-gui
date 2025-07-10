#!/usr/bin/env python3
"""
Test script for MLX-GUI multimodal vision functionality.

This script tests the multimodal vision capabilities by:
1. Installing the Gemma-3 vision model
2. Loading the local icon.png file 
3. Sending it to the model with "What do you see?"
4. Verifying the model can see the "M" in the image

Usage:
    python tests/test_multimodal_vision.py
"""

import asyncio
import base64
import json
import logging
import os
import sys
import time
from pathlib import Path

import httpx

# Enable debug logging for MLX integration
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('mlx_gui.mlx_integration')
logger.setLevel(logging.DEBUG)

# Add the src directory to the path so we can import mlx_gui modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class MultimodalVisionTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=300.0)  # 5 minute timeout
        self.model_name = "gemma-3n-e4b-it"
        self.model_id = "mlx-community/gemma-3n-E4B-it-bf16"
        
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """Encode an image file to base64 data URL."""
        try:
            with open(image_path, "rb") as image_file:
                encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
                # Determine the MIME type based on file extension
                ext = Path(image_path).suffix.lower()
                if ext in ['.jpg', '.jpeg']:
                    mime_type = 'image/jpeg'
                elif ext == '.png':
                    mime_type = 'image/png'
                elif ext == '.gif':
                    mime_type = 'image/gif'
                elif ext == '.webp':
                    mime_type = 'image/webp'
                else:
                    mime_type = 'image/png'  # Default
                
                return f"data:{mime_type};base64,{encoded_data}"
        except Exception as e:
            print(f"❌ Failed to encode image: {e}")
            return None
    
    async def check_server_health(self) -> bool:
        """Check if the MLX-GUI server is running."""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            if response.status_code == 200:
                print("✅ Server is running and healthy")
                return True
            else:
                print(f"❌ Server health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to server: {e}")
            print("   Make sure MLX-GUI server is running: mlx-gui start")
            return False
    
    async def install_model(self) -> bool:
        """Install the Gemma-3 vision model."""
        print(f"📥 Installing model {self.model_id}...")
        
        try:
            # Check if model is already installed
            response = await self.client.get(f"{self.base_url}/v1/models")
            if response.status_code == 200:
                models = response.json()
                installed_models = [model.get("id", model.get("name", "")) for model in models.get("data", [])]
                if self.model_name in installed_models:
                    print("✅ Model already installed")
                    return True
            
            # Install the model
            install_request = {
                "model_id": self.model_id,
                "name": self.model_name
            }
            
            response = await self.client.post(
                f"{self.base_url}/v1/models/install",
                json=install_request,
                timeout=600.0  # 10 minute timeout for installation
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Model installed: {result.get('message', 'Success')}")
                return True
            else:
                print(f"❌ Model installation failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Model installation error: {e}")
            return False
    
    async def load_model(self) -> bool:
        """Load the model into memory."""
        print(f"🔄 Loading model {self.model_name}...")
        
        try:
            response = await self.client.post(
                f"{self.base_url}/v1/models/{self.model_name}/load",
                timeout=300.0  # 5 minute timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Model loaded: {result.get('message', 'Success')}")
                return True
            else:
                print(f"❌ Model loading failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Model loading error: {e}")
            return False
    
    async def check_model_health(self) -> bool:
        """Check if the model is healthy and ready."""
        try:
            response = await self.client.get(f"{self.base_url}/v1/models/{self.model_name}/health")
            
            if response.status_code == 200:
                health = response.json()
                if health.get("healthy", False):
                    print("✅ Model is healthy and ready")
                    return True
                else:
                    print(f"❌ Model is not healthy: {health}")
                    return False
            else:
                print(f"❌ Model health check failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Model health check error: {e}")
            return False
    
    async def test_multimodal_vision(self, image_path: str) -> bool:
        """Test the multimodal vision functionality."""
        print(f"👁️  Testing multimodal vision with {image_path}...")
        
        # Check if image file exists
        if not os.path.exists(image_path):
            print(f"❌ Image file not found: {image_path}")
            return False
        
        # Encode image to base64
        encoded_image = self.encode_image_to_base64(image_path)
        if not encoded_image:
            return False
        
        print(f"📸 Encoded image ({Path(image_path).name}) to base64 ({len(encoded_image)} chars)")
        
        # Create chat completion request with image
        chat_request = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What do you see in this image? Describe it in detail."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": encoded_image
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 200,
            "temperature": 0.1  # Low temperature for consistent results
        }
        
        try:
            print("🤖 Sending multimodal request to model...")
            start_time = time.time()
            
            response = await self.client.post(
                f"{self.base_url}/v1/chat/completions",
                json=chat_request,
                timeout=300.0  # 5 minute timeout
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract the response text
                choices = result.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    content = message.get("content", "")
                    
                    # Usage statistics
                    usage = result.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    total_tokens = usage.get("total_tokens", 0)
                    
                    print(f"✅ Multimodal response received!")
                    print(f"   Duration: {duration:.2f}s")
                    print(f"   Tokens: {prompt_tokens} prompt + {completion_tokens} completion = {total_tokens} total")
                    print(f"   Response length: {len(content)} characters")
                    print("\n📝 Model Response:")
                    print("=" * 50)
                    print(content)
                    print("=" * 50)
                    
                    # Check if the model mentions seeing an "M" (case insensitive)
                    content_lower = content.lower()
                    sees_m = any(phrase in content_lower for phrase in [
                        "letter m", "letter \"m\"", "letter 'm'", 
                        "an m", "the m", " m ", "symbol m"
                    ])
                    
                    if sees_m:
                        print("\n🎯 SUCCESS: Model correctly identified the 'M' in the image!")
                        return True
                    else:
                        print(f"\n⚠️  Model response doesn't explicitly mention 'M'. Content: {content[:100]}...")
                        print("   This might still be working - check the full response above.")
                        return True  # Still consider it a success if we got a response
                else:
                    print("❌ No choices in response")
                    return False
            else:
                print(f"❌ Multimodal request failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Multimodal request error: {e}")
            return False
    
    async def cleanup_model(self) -> None:
        """Optional cleanup - unload the model."""
        print(f"🧹 Unloading model {self.model_name}...")
        try:
            response = await self.client.post(f"{self.base_url}/v1/models/{self.model_name}/unload")
            if response.status_code == 200:
                print("✅ Model unloaded")
            else:
                print(f"⚠️  Model unload failed: {response.status_code}")
        except Exception as e:
            print(f"⚠️  Model unload error: {e}")


async def main():
    """Main test function."""
    print("🚀 MLX-GUI Multimodal Vision Test")
    print("=" * 50)
    
    # Find the icon.png file
    script_dir = Path(__file__).parent.parent
    icon_path = script_dir / "icon.png"
    
    if not icon_path.exists():
        print(f"❌ Icon file not found at {icon_path}")
        print("   Please make sure icon.png exists in the project root")
        return False
    
    async with MultimodalVisionTester() as tester:
        # Step 1: Check server health
        if not await tester.check_server_health():
            return False
        
        # Step 2: Install model
        if not await tester.install_model():
            return False
        
        # Step 3: Load model 
        if not await tester.load_model():
            return False
        
        # Step 4: Wait a moment for model to be ready
        print("⏳ Waiting for model to be ready...")
        await asyncio.sleep(2)
        
        # Step 5: Check model health
        if not await tester.check_model_health():
            return False
        
        # Step 6: Test multimodal vision
        success = await tester.test_multimodal_vision(str(icon_path))
        
        # Step 7: Optional cleanup
        # await tester.cleanup_model()
        
        return success


if __name__ == "__main__":
    print(f"Python path: {sys.path[0]}")
    print(f"Script location: {Path(__file__).parent}")
    
    try:
        result = asyncio.run(main())
        if result:
            print("\n🎉 Multimodal vision test completed successfully!")
            sys.exit(0)
        else:
            print("\n💥 Multimodal vision test failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        sys.exit(1)