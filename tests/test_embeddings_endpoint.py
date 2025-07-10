#!/usr/bin/env python3
"""
Test script for the MLX-GUI embeddings endpoint.

This script tests the /v1/embeddings endpoint to ensure it works correctly
with the queuing system and MLX embedding models.
"""

import requests
import json
import sys
import time


def test_embeddings_endpoint():
    """Test the embeddings endpoint with a sample request."""
    
    # Base URL for the API
    BASE_URL = "http://localhost:8000"
    
    # Test data
    test_data = {
        "input": [
            "Hello, how are you?",
            "I am fine, thank you!",
            "This is a test of the embedding endpoint."
        ],
        "model": "qwen3-embedding-0-6b-4bit",  # Example: mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ
        "encoding_format": "float"
    }
    
    print("Testing MLX-GUI Embeddings Endpoint")
    print("=" * 50)
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ MLX-GUI server is not running or unhealthy")
            return False
        print("✅ Server is running and healthy")
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server: {e}")
        print("   Make sure MLX-GUI server is running on http://localhost:8000")
        return False
    
    # Test the embeddings endpoint
    print("\n📋 Testing embeddings endpoint...")
    print(f"   Model: {test_data['model']}")
    print(f"   Input texts: {len(test_data['input'])} items")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/v1/embeddings",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=60  # 1 minute timeout
        )
        end_time = time.time()
        
        print(f"   Response time: {end_time - start_time:.2f} seconds")
        print(f"   Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            # Validate response structure
            if "data" not in result:
                print("❌ Response missing 'data' field")
                return False
                
            if "usage" not in result:
                print("❌ Response missing 'usage' field")
                return False
                
            embeddings_data = result["data"]
            usage = result["usage"]
            
            print(f"✅ Embeddings generated successfully!")
            print(f"   Number of embeddings: {len(embeddings_data)}")
            print(f"   Embedding dimensions: {len(embeddings_data[0]['embedding']) if embeddings_data else 'N/A'}")
            print(f"   Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
            print(f"   Total tokens: {usage.get('total_tokens', 'N/A')}")
            
            # Validate each embedding
            for i, embedding_item in enumerate(embeddings_data):
                if "embedding" not in embedding_item:
                    print(f"❌ Embedding {i} missing 'embedding' field")
                    return False
                if "index" not in embedding_item:
                    print(f"❌ Embedding {i} missing 'index' field")
                    return False
                if not isinstance(embedding_item["embedding"], list):
                    print(f"❌ Embedding {i} is not a list")
                    return False
                if len(embedding_item["embedding"]) == 0:
                    print(f"❌ Embedding {i} is empty")
                    return False
            
            print("✅ All embeddings have valid structure")
            return True
            
        elif response.status_code == 404:
            print(f"❌ Model '{test_data['model']}' not found")
            print("   Install an embedding model first, for example:")
            print("   curl -X POST http://localhost:8000/v1/models/install \\")
            print("        -H 'Content-Type: application/json' \\")
            print("        -d '{\"model_id\": \"mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ\", \"name\": \"qwen3-embedding-0-6b-4bit\"}'")
            return False
            
        elif response.status_code == 503:
            print("❌ Service unavailable - model may be loading")
            print("   Try again in a few moments")
            return False
            
        else:
            print(f"❌ Request failed with status {response.status_code}")
            try:
                error_detail = response.json()
                print(f"   Error: {error_detail.get('detail', 'Unknown error')}")
            except:
                print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
        print("   The embedding request took too long to complete")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False
    except json.JSONDecodeError:
        print("❌ Invalid JSON response")
        print(f"   Response: {response.text}")
        return False


def test_embeddings_with_base64():
    """Test embeddings with base64 encoding format."""
    
    BASE_URL = "http://localhost:8000"
    
    test_data = {
        "input": "This is a test with base64 encoding.",
        "model": "qwen3-embedding-0-6b-4bit",
        "encoding_format": "base64"
    }
    
    print("\n📋 Testing embeddings with base64 encoding...")
    
    try:
        response = requests.post(
            f"{BASE_URL}/v1/embeddings",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            embedding_data = result["data"][0]["embedding"]
            
            # Check if it's base64 encoded (should be a string)
            if isinstance(embedding_data, str):
                print("✅ Base64 encoding works correctly")
                print(f"   Encoded length: {len(embedding_data)} characters")
                return True
            else:
                print("❌ Base64 encoding failed - result is not a string")
                return False
        else:
            print(f"❌ Base64 test failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Base64 test failed: {e}")
        return False


if __name__ == "__main__":
    print("MLX-GUI Embeddings Endpoint Test")
    print("This script tests the /v1/embeddings endpoint")
    print()
    
    # Run tests
    success1 = test_embeddings_endpoint()
    success2 = test_embeddings_with_base64()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("🎉 All tests passed!")
        print("   The embeddings endpoint is working correctly")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        print("   Check the output above for details")
        sys.exit(1)