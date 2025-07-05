"""
HuggingFace integration for MLX-GUI.
Handles model discovery, metadata extraction, and compatibility checking.
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path

from huggingface_hub import HfApi, list_models, model_info, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError
import requests

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a HuggingFace model."""
    id: str
    name: str
    author: str
    downloads: int
    likes: int
    created_at: str
    updated_at: str
    model_type: str
    library_name: Optional[str]
    pipeline_tag: Optional[str]
    tags: List[str]
    size_gb: Optional[float]
    mlx_compatible: bool
    has_mlx_version: bool
    mlx_repo_id: Optional[str]
    estimated_memory_gb: Optional[float]
    description: Optional[str]


class HuggingFaceClient:
    """Client for interacting with HuggingFace Hub."""
    
    def __init__(self, token: Optional[str] = None):
        self.api = HfApi(token=token)
        self.token = token
        self.mlx_tags = {
            "mlx",
            "mlx-lm", 
            "apple-silicon",
            "metal",
            "quantized"
        }
        
        # Parameter counts (in billions) for memory calculation
        self.param_patterns = {
            "0.5b": 0.5,
            "1b": 1.0,
            "1.8b": 1.8,
            "2.7b": 2.7,
            "3b": 3.0,
            "4b": 4.0,
            "6b": 6.0,
            "7b": 7.0,
            "8b": 8.0,
            "9b": 9.0,
            "11b": 11.0,
            "13b": 13.0,
            "14b": 14.0,
            "15b": 15.0,
            "20b": 20.0,
            "22b": 22.0,
            "24b": 24.0,
            "27b": 27.0,  # Gemma-3-27B
            "30b": 30.0,
            "32b": 32.0,
            "34b": 34.0,
            "40b": 40.0,
            "65b": 65.0,
            "70b": 70.0,
            "72b": 72.0,
            "110b": 110.0,
            "175b": 175.0,
            "235b": 235.0,  # Qwen3-235B
            "405b": 405.0,
        }
    
    def search_mlx_models(self, 
                         query: str = "",
                         limit: int = 50,
                         sort: str = "downloads") -> List[ModelInfo]:
        """
        Search for MLX-compatible models on HuggingFace.
        
        Args:
            query: Search query
            limit: Maximum number of results
            sort: Sort by 'downloads', 'likes', 'created', or 'updated'
            
        Returns:
            List of ModelInfo objects
        """
        try:
            # Get models using both library="mlx" and tags=["mlx"] approaches
            models_dict = {}  # Use dict to deduplicate by model ID
            
            # Approach 1: Models with library="mlx" 
            try:
                library_models = list_models(
                    search=query,
                    library="mlx",
                    limit=limit,
                    sort=sort,
                    direction=-1,
                    token=self.token
                )
                for model in library_models:
                    models_dict[model.id] = model
            except Exception as e:
                logger.warning(f"Library filter failed: {e}")
            
            # Approach 2: Models with mlx tag (broader set)
            try:
                tag_models = list_models(
                    search=query,
                    tags=["mlx"],
                    limit=limit,
                    sort=sort,
                    direction=-1,
                    token=self.token
                )
                for model in tag_models:
                    models_dict[model.id] = model
            except Exception as e:
                logger.warning(f"Tag filter failed: {e}")
            
            # Convert back to list and sort by downloads
            models = list(models_dict.values())
            models = sorted(models, key=lambda x: getattr(x, 'downloads', 0), reverse=True)[:limit]
            
            model_infos = []
            for model in models:
                try:
                    info = self._extract_model_info(model)
                    if info:
                        model_infos.append(info)
                except Exception as e:
                    logger.warning(f"Error processing model {model.id}: {e}")
                    continue
            
            return model_infos
            
        except Exception as e:
            if "429" in str(e) or "Too Many Requests" in str(e):
                logger.warning(f"Rate limited by HuggingFace: {e}")
                return []
            else:
                logger.error(f"Error searching MLX models: {e}")
                return []
    
    def search_models_by_author(self, author: str, limit: int = 20) -> List[ModelInfo]:
        """Search for models by a specific author."""
        try:
            models = list_models(
                author=author,
                limit=limit,
                sort="downloads",
                direction=-1
            )
            
            model_infos = []
            for model in models:
                try:
                    info = self._extract_model_info(model)
                    if info and info.mlx_compatible:
                        model_infos.append(info)
                except Exception as e:
                    logger.warning(f"Error processing model {model.id}: {e}")
                    continue
            
            return model_infos
            
        except Exception as e:
            logger.error(f"Error searching models by author {author}: {e}")
            return []
    
    def get_popular_mlx_models(self, limit: int = 20) -> List[ModelInfo]:
        """Get popular MLX models by downloads."""
        return self.search_mlx_models(limit=limit, sort="downloads")
    
    def get_recent_mlx_models(self, limit: int = 20) -> List[ModelInfo]:
        """Get recently updated MLX models."""
        return self.search_mlx_models(limit=limit, sort="updated")
    
    def get_model_details(self, model_id: str) -> Optional[ModelInfo]:
        """Get detailed information about a specific model."""
        try:
            model = model_info(model_id)
            return self._extract_model_info(model)
        except (RepositoryNotFoundError, RevisionNotFoundError) as e:
            logger.error(f"Model {model_id} not found: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting model details for {model_id}: {e}")
            return None
    
    def _extract_model_info(self, model) -> Optional[ModelInfo]:
        """Extract ModelInfo from HuggingFace model object."""
        try:
            # Parse model ID
            parts = model.id.split("/")
            author = parts[0] if len(parts) > 1 else "unknown"
            name = parts[-1]
            
            # Check if model is MLX compatible
            tags = model.tags or []
            mlx_compatible = any(tag.lower() in self.mlx_tags for tag in tags)
            has_mlx_version = "mlx" in tags
            
            # If not explicitly MLX tagged, check if there's an MLX version
            mlx_repo_id = None
            if not mlx_compatible:
                # Check if there's a corresponding MLX version
                potential_mlx_id = f"mlx-community/{name}"
                try:
                    mlx_model = model_info(potential_mlx_id)
                    if mlx_model:
                        mlx_repo_id = potential_mlx_id
                        has_mlx_version = True
                        mlx_compatible = True
                except:
                    pass
            
            # Estimate model size and memory requirements
            size_gb = self._estimate_model_size(model)
            # Size calculation already includes MLX overhead, use as-is
            estimated_memory_gb = size_gb
            
            # Get model type
            model_type = self._determine_model_type(model, tags)
            
            return ModelInfo(
                id=model.id,
                name=name,
                author=author,
                downloads=getattr(model, 'downloads', 0),
                likes=getattr(model, 'likes', 0),
                created_at=model.created_at.isoformat() if hasattr(model, 'created_at') and model.created_at else None,
                updated_at=model.last_modified.isoformat() if hasattr(model, 'last_modified') and model.last_modified else None,
                model_type=model_type,
                library_name=getattr(model, 'library_name', None),
                pipeline_tag=getattr(model, 'pipeline_tag', None),
                tags=tags,
                size_gb=size_gb,
                mlx_compatible=mlx_compatible,
                has_mlx_version=has_mlx_version,
                mlx_repo_id=mlx_repo_id,
                estimated_memory_gb=estimated_memory_gb,
                description=getattr(model, 'description', None)
            )
            
        except Exception as e:
            logger.error(f"Error extracting model info: {e}")
            return None
    
    def _estimate_model_size(self, model) -> Optional[float]:
        """Calculate model memory requirements from parameter count and quantization."""
        try:
            model_name = model.id.lower()
            
            # Find parameter count from model name (longest match wins)
            param_count_billions = None
            best_match_length = 0
            for pattern, param_count in self.param_patterns.items():
                if pattern in model_name and len(pattern) > best_match_length:
                    param_count_billions = param_count
                    best_match_length = len(pattern)
            
            if param_count_billions:
                # Determine bits per parameter from quantization
                bits_per_param = 16  # Default FP16
                
                if "4bit" in model_name or "4-bit" in model_name or "qat-4bit" in model_name:
                    bits_per_param = 4
                elif "6bit" in model_name or "6-bit" in model_name:
                    bits_per_param = 6
                elif "8bit" in model_name or "8-bit" in model_name or "int8" in model_name:
                    bits_per_param = 8
                elif "int4" in model_name:
                    bits_per_param = 4
                elif "bf16" in model_name or "fp16" in model_name:
                    bits_per_param = 16
                elif "fp32" in model_name:
                    bits_per_param = 32
                
                # Calculate base memory: params * bits_per_param / 8 bits_per_byte
                base_memory_gb = (param_count_billions * 1e9 * bits_per_param) / (8 * 1024**3)
                
                # Add MLX overhead (25% for inference, activations, etc.)
                total_memory_gb = base_memory_gb * 1.25
                
                logger.debug(f"Model {model.id}: {param_count_billions}B params, {bits_per_param}-bit = {total_memory_gb:.1f}GB")
                return total_memory_gb
            
            # Fallback: estimate from file size (less accurate)
            try:
                # Use model_info instead of listing files individually (more efficient)
                model_info_data = self.api.model_info(model.id)
                if hasattr(model_info_data, 'safetensors') and model_info_data.safetensors:
                    # Use safetensors metadata if available
                    total_size = sum(
                        file_data.get('total_size', 0) 
                        for file_data in model_info_data.safetensors.values()
                    )
                    if total_size > 0:
                        file_size_gb = total_size / (1024**3)
                        return file_size_gb * 1.5
                
                # Last resort: estimate default size for unknown models
                return 4.0  # 4GB default
                    
            except Exception as e:
                logger.debug(f"Could not get file sizes for {model.id}: {e}")
            
            return None
            
        except Exception as e:
            logger.debug(f"Error estimating model size for {model.id}: {e}")
            return None
    
    def _determine_model_type(self, model, tags: List[str]) -> str:
        """Determine model type from tags and metadata."""
        tags_lower = [tag.lower() for tag in tags]
        
        # Check for multimodal capabilities
        if any(tag in tags_lower for tag in ['multimodal', 'vision', 'image-text', 'vlm']):
            return 'multimodal'
        
        # Check for vision models
        if any(tag in tags_lower for tag in ['computer-vision', 'image-classification', 'object-detection']):
            return 'vision'
        
        # Check for audio models
        if any(tag in tags_lower for tag in ['audio', 'speech', 'automatic-speech-recognition']):
            return 'audio'
        
        # Default to text for language models
        pipeline_tag = getattr(model, 'pipeline_tag', '')
        if pipeline_tag in ['text-generation', 'text2text-generation', 'conversational']:
            return 'text'
        
        # Check tags for text generation
        if any(tag in tags_lower for tag in ['text-generation', 'language-modeling', 'causal-lm']):
            return 'text'
        
        return 'text'  # Default
    
    def get_model_categories(self) -> Dict[str, List[str]]:
        """Get categorized lists of popular MLX models."""
        categories = {
            'Popular Text Models': [],
            'Vision Models': [],
            'Multimodal Models': [],
            'Code Models': [],
            'Chat Models': [],
            'Small Models (< 10GB)': [],
            'Large Models (> 50GB)': []
        }
        
        try:
            # Get popular models
            popular_models = self.get_popular_mlx_models(limit=100)
            
            for model in popular_models:
                # Add to appropriate categories
                if model.model_type == 'vision':
                    categories['Vision Models'].append(model.id)
                elif model.model_type == 'multimodal':
                    categories['Multimodal Models'].append(model.id)
                
                # Check for code models
                if any(tag in model.tags for tag in ['code', 'coding', 'programming']):
                    categories['Code Models'].append(model.id)
                
                # Check for chat models
                if any(tag in model.tags for tag in ['chat', 'conversational', 'instruct']):
                    categories['Chat Models'].append(model.id)
                
                # Size-based categories
                if model.size_gb and model.size_gb < 10:
                    categories['Small Models (< 10GB)'].append(model.id)
                elif model.size_gb and model.size_gb > 50:
                    categories['Large Models (> 50GB)'].append(model.id)
                
                # Add to popular text models if not in other categories
                if model.model_type == 'text' and not any(model.id in cat for cat in categories.values()):
                    categories['Popular Text Models'].append(model.id)
            
            # Limit each category to top 10
            for category in categories:
                categories[category] = categories[category][:10]
            
            return categories
            
        except Exception as e:
            logger.error(f"Error getting model categories: {e}")
            return categories
    
    def search_compatible_models(self, query: str, max_memory_gb: float) -> List[ModelInfo]:
        """Search for models compatible with available memory."""
        models = self.search_mlx_models(query, limit=100)
        
        compatible_models = []
        for model in models:
            if model.estimated_memory_gb and model.estimated_memory_gb <= max_memory_gb:
                compatible_models.append(model)
        
        # Sort by popularity (downloads)
        compatible_models.sort(key=lambda x: x.downloads, reverse=True)
        
        return compatible_models[:20]  # Return top 20


# Global HuggingFace client instance
_hf_client: Optional[HuggingFaceClient] = None


def get_huggingface_client(token: Optional[str] = None) -> HuggingFaceClient:
    """Get the global HuggingFace client instance."""
    global _hf_client
    if _hf_client is None or (token and _hf_client.token != token):
        # Check for token in environment if not provided
        if not token:
            import os
            token = os.getenv('HUGGINGFACE_HUB_TOKEN') or os.getenv('HF_TOKEN')
        
        _hf_client = HuggingFaceClient(token=token)
    return _hf_client