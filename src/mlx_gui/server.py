"""
FastAPI server for MLX-GUI REST API.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Depends, HTTPException, status, Header, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional, Union, Annotated
import json
import tempfile
import os
import uuid

from mlx_gui.database import get_db_session, get_database_manager
from mlx_gui.models import Model, AppSettings
from mlx_gui.system_monitor import get_system_monitor
from mlx_gui.huggingface_integration import get_huggingface_client
from mlx_gui.model_manager import get_model_manager
from mlx_gui.mlx_integration import GenerationConfig, get_inference_engine
from mlx_gui.inference_queue_manager import get_inference_manager, QueuedRequest
from mlx_gui.queued_inference import queued_generate_text, queued_generate_text_stream, queued_transcribe_audio, queued_generate_speech
from mlx_gui import __version__

logger = logging.getLogger(__name__)

# API Key validation (accepts any key for OpenAI compatibility)
security = HTTPBearer(auto_error=False)

def validate_api_key(
    authorization: Optional[HTTPAuthorizationCredentials] = Depends(security),
    x_api_key: Optional[str] = Header(None, alias="x-api-key")
) -> Optional[str]:
    """
    Validate API key for OpenAI compatibility.
    Accepts any key via Authorization header or x-api-key header.
    """
    # Check Authorization: Bearer <token>
    if authorization:
        return authorization.credentials
    
    # Check x-api-key header
    if x_api_key:
        return x_api_key
    
    # For OpenAI compatibility, we accept any key, so return None if no key provided
    # This allows both authenticated and unauthenticated access
    return None


# Pydantic models for OpenAI compatibility
from typing import Union, Any, Dict

class ChatMessageContent(BaseModel):
    """Content part for multimodal messages."""
    type: str  # "text" or "image_url"
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None

class ChatMessage(BaseModel):
    role: Optional[str] = None  # "system", "user", "assistant"
    content: Optional[Union[str, List[ChatMessageContent]]] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 8192
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = 0
    repetition_penalty: Optional[float] = 1.0
    seed: Optional[int] = None
    stream: Optional[bool] = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: dict  # Use dict instead of ChatMessage for flexibility
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]


class ModelInstallRequest(BaseModel):
    model_id: str
    name: Optional[str] = None


# Audio API models
class AudioTranscriptionRequest(BaseModel):
    """Request model for audio transcription."""
    model: str = "whisper-1"
    language: Optional[str] = None
    prompt: Optional[str] = None
    response_format: Optional[str] = "json"  # json, text, srt, verbose_json, vtt
    temperature: Optional[float] = 0.0


class AudioTranscriptionResponse(BaseModel):
    """Response model for audio transcription."""
    text: str


class AudioSpeechRequest(BaseModel):
    """Request model for text-to-speech."""
    model: str = "tts-1"  # tts-1, tts-1-hd
    input: str
    voice: str = "alloy"  # alloy, echo, fable, onyx, nova, shimmer
    response_format: Optional[str] = "mp3"  # mp3, opus, aac, flac
    speed: Optional[float] = 1.0  # 0.25 to 4.0


def _get_chat_template_from_hf(model_id: str) -> str:
    """Fetch chat template from HuggingFace model card."""
    try:
        from huggingface_hub import hf_hub_download
        import json
        
        # Download tokenizer_config.json which contains the chat template
        tokenizer_config_path = hf_hub_download(
            repo_id=model_id,
            filename="tokenizer_config.json",
            local_files_only=False
        )
        
        with open(tokenizer_config_path, 'r') as f:
            tokenizer_config = json.load(f)
            
        chat_template = tokenizer_config.get("chat_template")
        if chat_template:
            logger.info(f"Retrieved chat template from HF for {model_id}")
            return chat_template
            
    except Exception as e:
        logger.debug(f"Could not fetch chat template from HF for {model_id}: {e}")
    
    return None


def _format_chat_prompt(messages: List[ChatMessage], model_name: str = None) -> str:
    """Convert chat messages to a formatted prompt using the model's chat template."""
    
    # Convert our messages to the format expected by chat templates
    chat_messages = []
    for message in messages:
        content = message.content
        
        # Handle multimodal content - extract text
        if isinstance(content, list):
            text_parts = []
            images = []
            for part in content:
                if part.type == "text" and part.text:
                    text_parts.append(part.text)
                elif part.type == "image_url" and part.image_url:
                    # For now, we'll include a placeholder for the image
                    # MLX models will need to handle the base64 image data
                    images.append(part.image_url.get("url", ""))
            
            # Combine text parts
            text_content = " ".join(text_parts)
            
            # Add image information if present
            if images:
                image_info = f" [Image data: {len(images)} image(s)]"
                text_content += image_info
                
            content = text_content
        
        chat_messages.append({"role": message.role, "content": content})
    
    # Try to use the model's tokenizer chat template if available
    if model_name:
        try:
            from mlx_gui.model_manager import get_model_manager
            model_manager = get_model_manager()
            loaded_model = model_manager.get_model_for_inference(model_name)
            
            if loaded_model and hasattr(loaded_model.mlx_wrapper, 'tokenizer') and loaded_model.mlx_wrapper.tokenizer:
                tokenizer = loaded_model.mlx_wrapper.tokenizer
                
                # Use the tokenizer's chat template
                if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
                    try:
                        formatted_prompt = tokenizer.apply_chat_template(
                            chat_messages, 
                            tokenize=False, 
                            add_generation_prompt=True
                        )
                        logger.debug(f"Used tokenizer chat template for {model_name}")
                        return formatted_prompt
                    except Exception as e:
                        logger.warning(f"Failed to apply chat template for {model_name}: {e}")
        except Exception as e:
            logger.warning(f"Could not get tokenizer for {model_name}: {e}")
        
        # Try to fetch template from HuggingFace if tokenizer failed
        try:
            # Get the HuggingFace model ID from the database
            from mlx_gui.database import get_database_manager
            from mlx_gui.models import Model
            
            db_manager = get_database_manager()
            with db_manager.get_session() as session:
                model_record = session.query(Model).filter(Model.name == model_name).first()
                if model_record and model_record.huggingface_id:
                    hf_template = _get_chat_template_from_hf(model_record.huggingface_id)
                    if hf_template:
                        # Try to apply the template using transformers
                        try:
                            from transformers import AutoTokenizer
                            # Create a temporary tokenizer with the fetched template
                            temp_tokenizer = AutoTokenizer.from_pretrained(
                                model_record.huggingface_id,
                                trust_remote_code=True
                            )
                            if hasattr(temp_tokenizer, 'apply_chat_template'):
                                formatted_prompt = temp_tokenizer.apply_chat_template(
                                    chat_messages,
                                    tokenize=False,
                                    add_generation_prompt=True
                                )
                                logger.info(f"Used HF chat template for {model_name}")
                                return formatted_prompt
                        except Exception as e:
                            logger.warning(f"Failed to apply HF chat template for {model_name}: {e}")
        except Exception as e:
            logger.debug(f"Could not fetch HF template for {model_name}: {e}")
    
    # Fallback to manual formatting with model-specific templates
    prompt_parts = []
    
    # Detect model type and use appropriate template
    is_gemma = model_name and "gemma" in model_name.lower()
    is_phi = model_name and "phi" in model_name.lower()
    is_qwen = model_name and "qwen" in model_name.lower()
    
    for message in chat_messages:
        role = message["role"]
        content = message["content"]
        
        # Apply model-specific formatting
        if is_gemma:
            # Gemma format
            if role == "system":
                prompt_parts.append(f"<bos><start_of_turn>system\n{content}<end_of_turn>")
            elif role == "user":
                prompt_parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
            elif role == "assistant":
                prompt_parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")
        elif is_phi:
            # Phi format (simpler)
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        else:
            # ChatML format (default for Qwen and others)
            if role == "system":
                prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
    
    # Add final generation prompt
    if is_gemma:
        prompt_parts.append("<start_of_turn>model\n")
    elif is_phi:
        prompt_parts.append("Assistant:")
    else:
        prompt_parts.append("<|im_start|>assistant")
    
    formatted_prompt = "\n".join(prompt_parts)
    logger.info(f"Used fallback template for {model_name}: {'Gemma' if is_gemma else 'Phi' if is_phi else 'ChatML'}")
    return formatted_prompt


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    logger.info("Starting MLX-GUI server...")
    
    # Initialize database
    db_manager = get_database_manager()
    logger.info(f"Database initialized at: {db_manager.database_path}")
    
    yield
    
    # Cleanup
    try:
        # Kill everything immediately
        from mlx_gui.model_manager import shutdown_model_manager
        from mlx_gui.inference_queue_manager import shutdown_inference_manager
        shutdown_inference_manager()
        shutdown_model_manager()
        db_manager.close()
    except:
        pass


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="MLX-GUI API",
        description="A lightweight RESTful wrapper around Apple's MLX engine",
        version=__version__,
        lifespan=lifespan,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Exception handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"}
        )
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with basic server info."""
        return {
            "name": "MLX-GUI API",
            "version": __version__,
            "status": "running"
        }
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}
    
    # API v1 routes
    @app.get("/v1/manager/models")
    async def list_models_internal(db: Session = Depends(get_db_session)):
        """List all models (internal format)."""
        models = db.query(Model).all()
        return {
            "models": [
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
                    "author": model.huggingface_id.split("/")[0] if model.huggingface_id and "/" in model.huggingface_id else "unknown",
                }
                for model in models
            ]
        }
    
    @app.get("/v1/models/{model_name}")
    async def get_model(model_name: str, db: Session = Depends(get_db_session)):
        """Get specific model details."""
        model = db.query(Model).filter(Model.name == model_name).first()
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_name}' not found"
            )
        
        return {
            "id": model.id,
            "name": model.name,
            "path": model.path,
            "version": model.version,
            "type": model.model_type,
            "status": model.status,
            "memory_required_gb": model.memory_required_gb,
            "use_count": model.use_count,
            "last_used_at": model.last_used_at.isoformat() if model.last_used_at else None,
            "created_at": model.created_at.isoformat() if model.created_at else None,
            "updated_at": model.updated_at.isoformat() if model.updated_at else None,
            "error_message": model.error_message,
            "metadata": model.get_metadata(),
        }
    
    @app.post("/v1/models/{model_name}/load")
    async def load_model(
        model_name: str,
        priority: int = 0,
        db: Session = Depends(get_db_session)
    ):
        """Load a model."""
        model_record = db.query(Model).filter(Model.name == model_name).first()
        if not model_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_name}' not found"
            )
        
        try:
            model_manager = get_model_manager()
            
            # Check if already loaded
            if model_name in model_manager._loaded_models:
                return {
                    "message": f"Model '{model_name}' is already loaded",
                    "status": "loaded"
                }
            
            # Check system compatibility
            system_monitor = get_system_monitor()
            can_load, compatibility_message = system_monitor.check_model_compatibility(
                model_record.memory_required_gb
            )
            
            if not can_load:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=compatibility_message
                )
            
            # Initiate loading
            success = await model_manager.load_model_async(
                model_name=model_name,
                model_path=model_record.path,
                priority=priority
            )
            
            if success:
                return {
                    "message": f"Model '{model_name}' loaded successfully",
                    "status": "loaded"
                }
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to load model '{model_name}'"
                )
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error loading model: {str(e)}"
            )
    
    @app.post("/v1/models/{model_name}/unload")
    async def unload_model(
        model_name: str,
        db: Session = Depends(get_db_session)
    ):
        """Unload a model."""
        model_record = db.query(Model).filter(Model.name == model_name).first()
        if not model_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_name}' not found"
            )
        
        try:
            model_manager = get_model_manager()
            success = model_manager.unload_model(model_name)
            
            if success:
                return {
                    "message": f"Model '{model_name}' unloaded successfully",
                    "status": "unloaded"
                }
            else:
                return {
                    "message": f"Model '{model_name}' was not loaded",
                    "status": "not_loaded"
                }
                
        except Exception as e:
            logger.error(f"Error unloading model {model_name}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error unloading model: {str(e)}"
            )
    
    @app.delete("/v1/models/{model_name}")
    async def delete_model(
        model_name: str,
        remove_files: bool = True,
        db: Session = Depends(get_db_session)
    ):
        """Delete a model from the database and optionally remove files."""
        model_record = db.query(Model).filter(Model.name == model_name).first()
        if not model_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_name}' not found"
            )
        
        try:
            # First unload if loaded
            model_manager = get_model_manager()
            model_manager.unload_model(model_name)
            
            # Remove from database
            db.delete(model_record)
            db.commit()
            
            # Optionally remove downloaded files
            if remove_files and model_record.path:
                try:
                    import shutil
                    import os
                    from pathlib import Path
                    
                    # If it's a HuggingFace cache path, remove the entire model directory
                    if ".cache" in model_record.path and "models--" in model_record.path:
                        # Extract the model directory from the path
                        cache_path = Path(model_record.path)
                        if cache_path.exists():
                            # Find the models--* directory
                            for parent in cache_path.parents:
                                if parent.name.startswith("models--"):
                                    if parent.exists():
                                        shutil.rmtree(parent)
                                        logger.info(f"Removed model files at {parent}")
                                    break
                    elif os.path.exists(model_record.path):
                        # Remove local model directory
                        if os.path.isdir(model_record.path):
                            shutil.rmtree(model_record.path)
                        else:
                            os.remove(model_record.path)
                        logger.info(f"Removed model files at {model_record.path}")
                        
                except Exception as file_error:
                    logger.warning(f"Could not remove model files: {file_error}")
                    # Don't fail the deletion if file removal fails
            
            return {
                "message": f"Model '{model_name}' deleted successfully",
                "removed_files": remove_files,
                "status": "deleted"
            }
                
        except Exception as e:
            logger.error(f"Error deleting model {model_name}: {e}")
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error deleting model: {str(e)}"
            )
    
    @app.get("/v1/models/{model_name}/health")
    async def model_health(
        model_name: str,
        db: Session = Depends(get_db_session)
    ):
        """Check model health status."""
        model = db.query(Model).filter(Model.name == model_name).first()
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_name}' not found"
            )
        
        return {
            "model": model_name,
            "status": model.status,
            "healthy": model.status == "loaded",
            "last_used": model.last_used_at.isoformat() if model.last_used_at else None,
        }
    
    @app.post("/v1/models/{model_name}/generate")
    async def generate_text(
        model_name: str,
        request_data: dict,
        db: Session = Depends(get_db_session)
    ):
        """Generate text using a model."""
        model_record = db.query(Model).filter(Model.name == model_name).first()
        if not model_record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{model_name}' not found"
            )
        
        try:
            model_manager = get_model_manager()
            
            # Check if model is loaded
            loaded_model = model_manager.get_model_for_inference(model_name)
            if not loaded_model:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Model '{model_name}' is not loaded. Load it first with POST /v1/models/{model_name}/load"
                )
            
            # Extract generation parameters
            prompt = request_data.get("prompt", "")
            if not prompt:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Prompt is required"
                )
            
            # Create generation config
            config = GenerationConfig(
                max_tokens=request_data.get("max_tokens", 100),
                temperature=request_data.get("temperature", 0.0),
                top_p=request_data.get("top_p", 1.0),
                top_k=request_data.get("top_k", 0),
                repetition_penalty=request_data.get("repetition_penalty", 1.0),
                repetition_context_size=request_data.get("repetition_context_size", 20),
                seed=request_data.get("seed")
            )
            
            # Generate text with transparent queuing
            result = await queued_generate_text(model_name, prompt, config)
            
            return {
                "model": model_name,
                "prompt": result.prompt,
                "text": result.text,
                "usage": {
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "total_tokens": result.total_tokens
                },
                "timing": {
                    "generation_time_seconds": result.generation_time_seconds,
                    "tokens_per_second": result.tokens_per_second
                }
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error generating text with model {model_name}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error generating text: {str(e)}"
            )
    
    @app.get("/v1/system/status")
    async def system_status(db: Session = Depends(get_db_session)):
        """Get system status including memory usage."""
        system_monitor = get_system_monitor()
        system_summary = system_monitor.get_system_summary()
        
        model_manager = get_model_manager()
        manager_status = model_manager.get_system_status()
        
        return {
            "status": "running",
            "system": system_summary,
            "model_manager": manager_status,
            "mlx_compatible": system_summary["mlx_compatible"]
        }
    
    @app.get("/v1/system/version")
    async def get_version():
        """Get application version information."""
        from mlx_gui import __version__, __author__, __description__
        return {
            "version": __version__,
            "author": __author__,
            "description": __description__,
            "name": "MLX-GUI"
        }
    
    @app.get("/v1/settings")
    async def get_settings(db: Session = Depends(get_db_session)):
        """Get application settings."""
        settings = db.query(AppSettings).all()
        return {
            setting.key: setting.get_typed_value()
            for setting in settings
        }
    
    @app.put("/v1/settings/{key}")
    async def update_setting(
        key: str,
        value: dict,
        db: Session = Depends(get_db_session)
    ):
        """Update a setting value."""
        setting = db.query(AppSettings).filter(AppSettings.key == key).first()
        if not setting:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Setting '{key}' not found"
            )
        
        # Extract value from request body
        new_value = value.get("value")
        if new_value is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Value is required"
            )
        
        setting.set_typed_value(new_value)
        db.commit()
        
        return {
            "key": key,
            "value": setting.get_typed_value(),
            "updated": True
        }
    
    # HuggingFace model discovery endpoints
    @app.get("/v1/discover/models")
    async def discover_models(
        query: str = "",
        limit: int = 20,
        sort: str = "downloads"
    ):
        """Discover MLX-compatible models from HuggingFace."""
        try:
            hf_client = get_huggingface_client()
            models = hf_client.search_mlx_models(query=query, limit=limit, sort=sort)
            
            return {
                "models": [
                    {
                        "id": model.id,
                        "name": model.name,
                        "author": model.author,
                        "downloads": model.downloads,
                        "likes": model.likes,
                        "model_type": model.model_type,
                        "size_gb": model.size_gb,
                        "estimated_memory_gb": model.estimated_memory_gb,
                        "mlx_compatible": model.mlx_compatible,
                        "has_mlx_version": model.has_mlx_version,
                        "mlx_repo_id": model.mlx_repo_id,
                        "tags": model.tags,
                        "description": model.description,
                        "updated_at": model.updated_at
                    }
                    for model in models
                ],
                "total": len(models)
            }
        except Exception as e:
            logger.error(f"Error discovering models: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error discovering models from HuggingFace"
            )
    
    @app.get("/v1/discover/popular")
    async def discover_popular_models(limit: int = 20):
        """Get popular MLX models."""
        try:
            hf_client = get_huggingface_client()
            models = hf_client.get_popular_mlx_models(limit=limit)
            
            return {
                "models": [
                    {
                        "id": model.id,
                        "name": model.name,
                        "author": model.author,
                        "downloads": model.downloads,
                        "likes": model.likes,
                        "model_type": model.model_type,
                        "size_gb": model.size_gb,
                        "estimated_memory_gb": model.estimated_memory_gb,
                        "mlx_compatible": model.mlx_compatible,
                        "description": model.description
                    }
                    for model in models
                ],
                "total": len(models)
            }
        except Exception as e:
            logger.error(f"Error getting popular models: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error getting popular models"
            )
    
    @app.get("/v1/discover/categories")
    async def get_model_categories():
        """Get categorized model lists."""
        try:
            hf_client = get_huggingface_client()
            categories = hf_client.get_model_categories()
            return {"categories": categories}
        except Exception as e:
            logger.error(f"Error getting model categories: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error getting model categories"
            )
    
    @app.get("/v1/discover/vision")
    async def discover_vision_models(query: str = "", limit: int = 10):
        """Discover vision/multimodal models using HuggingFace pipeline filters."""
        try:
            hf_client = get_huggingface_client()
            models = hf_client.search_vision_models(query=query, limit=limit)
            
            return {
                "models": [
                    {
                        "id": model.id,
                        "name": model.name,
                        "author": model.author,
                        "downloads": model.downloads,
                        "likes": model.likes,
                        "model_type": model.model_type,
                        "size_gb": model.size_gb,
                        "estimated_memory_gb": model.estimated_memory_gb,
                        "mlx_compatible": model.mlx_compatible,
                        "has_mlx_version": model.has_mlx_version,
                        "mlx_repo_id": model.mlx_repo_id,
                        "tags": model.tags,
                        "description": model.description,
                        "updated_at": model.updated_at
                    }
                    for model in models
                ],
                "total": len(models)
            }
        except Exception as e:
            logger.error(f"Error discovering vision models: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error discovering vision models"
            )
    
    @app.get("/v1/discover/compatible")
    async def discover_compatible_models(
        query: str = "",
        max_memory_gb: Optional[float] = None
    ):
        """Discover models compatible with current system."""
        try:
            # Get system memory if not specified
            if max_memory_gb is None:
                system_monitor = get_system_monitor()
                memory_info = system_monitor.get_memory_info()
                max_memory_gb = memory_info.total_gb * 0.8  # Use 80% of total RAM
            
            hf_client = get_huggingface_client()
            models = hf_client.search_compatible_models(query, max_memory_gb)
            
            return {
                "models": [
                    {
                        "id": model.id,
                        "name": model.name,
                        "author": model.author,
                        "downloads": model.downloads,
                        "likes": model.likes,
                        "model_type": model.model_type,
                        "size_gb": model.size_gb,
                        "estimated_memory_gb": model.estimated_memory_gb,
                        "mlx_compatible": model.mlx_compatible,
                        "description": model.description,
                        "memory_fit": f"{model.estimated_memory_gb:.1f}GB required, {max_memory_gb:.1f}GB available"
                    }
                    for model in models
                ],
                "max_memory_gb": max_memory_gb,
                "total": len(models)
            }
        except Exception as e:
            logger.error(f"Error discovering compatible models: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error discovering compatible models"
            )
    
    @app.get("/v1/discover/models/{model_id:path}")
    async def get_model_details(model_id: str):
        """Get detailed information about a specific HuggingFace model."""
        try:
            hf_client = get_huggingface_client()
            model = hf_client.get_model_details(model_id)
            
            if not model:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model '{model_id}' not found on HuggingFace"
                )
            
            # Check system compatibility
            system_monitor = get_system_monitor()
            can_load, compatibility_message = system_monitor.check_model_compatibility(
                model.estimated_memory_gb or 0
            )
            
            return {
                "id": model.id,
                "name": model.name,
                "author": model.author,
                "downloads": model.downloads,
                "likes": model.likes,
                "created_at": model.created_at,
                "updated_at": model.updated_at,
                "model_type": model.model_type,
                "library_name": model.library_name,
                "pipeline_tag": model.pipeline_tag,
                "tags": model.tags,
                "size_gb": model.size_gb,
                "estimated_memory_gb": model.estimated_memory_gb,
                "mlx_compatible": model.mlx_compatible,
                "has_mlx_version": model.has_mlx_version,
                "mlx_repo_id": model.mlx_repo_id,
                "description": model.description,
                "system_compatible": can_load,
                "compatibility_message": compatibility_message
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting model details for {model_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error getting model details"
            )
    
    # OpenAI-compatible endpoints
    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: ChatCompletionRequest,
        db: Session = Depends(get_db_session),
        api_key: Optional[str] = Depends(validate_api_key)
    ):
        """OpenAI-compatible chat completions endpoint."""
        try:
            # Log API key usage (for debugging)
            if api_key:
                logger.debug(f"API key provided: {api_key[:8]}...")
            else:
                logger.debug("No API key provided")
            
            # Check if model exists in database
            model_record = db.query(Model).filter(Model.name == request.model).first()
            if not model_record:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model '{request.model}' not found. Install it first with POST /v1/models/install"
                )
            
            model_manager = get_model_manager()
            
            # Check if model is loaded, auto-load if not
            loaded_model = model_manager.get_model_for_inference(request.model)
            if not loaded_model:
                logger.info(f"Model {request.model} not loaded, attempting to load...")
                success = await model_manager.load_model_async(
                    model_name=request.model,
                    model_path=model_record.path,
                    priority=10  # High priority for chat requests
                )
                if not success:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail=f"Failed to load model '{request.model}'"
                    )
            
            # Add default system prompt if none provided
            messages = request.messages
            has_system_message = any(msg.role == "system" for msg in messages)
            
            if not has_system_message:
                # Add a helpful default system message
                default_system = ChatMessage(
                    role="system",
                    content="You are a helpful AI assistant. Provide clear, accurate, and concise responses."
                )
                messages = [default_system] + list(messages)
            
            # Convert chat messages to prompt
            prompt = _format_chat_prompt(messages, request.model)
            
            # Enforce server-side maximum token limit
            MAX_TOKENS_LIMIT = 16384  # 16k max
            if request.max_tokens > MAX_TOKENS_LIMIT:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"max_tokens cannot exceed {MAX_TOKENS_LIMIT}, requested {request.max_tokens}"
                )
            
            # Create generation config
            config = GenerationConfig(
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
                seed=request.seed
            )
            
            import time
            import uuid
            
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"
            created_time = int(time.time())
            
            # Handle streaming vs non-streaming
            if request.stream:
                # Streaming response
                async def generate_stream():
                    """Generate streaming response chunks."""
                    # First chunk with role
                    first_chunk = ChatCompletionStreamResponse(
                        id=completion_id,
                        created=created_time,
                        model=request.model,
                        choices=[
                            ChatCompletionStreamChoice(
                                index=0,
                                delta={"role": "assistant"},
                                finish_reason=None
                            )
                        ]
                    )
                    yield f"data: {first_chunk.model_dump_json()}\n\n"
                    
                    # Stream the generation with transparent queuing
                    async for chunk in queued_generate_text_stream(request.model, prompt, config):
                        if chunk:
                            stream_chunk = ChatCompletionStreamResponse(
                                id=completion_id,
                                created=created_time,
                                model=request.model,
                                choices=[
                                    ChatCompletionStreamChoice(
                                        index=0,
                                        delta={"content": chunk},
                                        finish_reason=None
                                    )
                                ]
                            )
                            yield f"data: {stream_chunk.model_dump_json()}\n\n"
                    
                    # Final chunk with finish_reason
                    final_chunk = ChatCompletionStreamResponse(
                        id=completion_id,
                        created=created_time,
                        model=request.model,
                        choices=[
                            ChatCompletionStreamChoice(
                                index=0,
                                delta={},
                                finish_reason="stop"
                            )
                        ]
                    )
                    yield f"data: {final_chunk.model_dump_json()}\n\n"
                    yield "data: [DONE]\n\n"
                
                return StreamingResponse(
                    generate_stream(),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
                )
            else:
                # Non-streaming response with transparent queuing
                result = await queued_generate_text(request.model, prompt, config)
                
                response = ChatCompletionResponse(
                    id=completion_id,
                    created=created_time,
                    model=request.model,
                    choices=[
                        ChatCompletionChoice(
                            index=0,
                            message=ChatMessage(
                                role="assistant",
                                content=result.text
                            ),
                            finish_reason="stop"
                        )
                    ],
                    usage=ChatCompletionUsage(
                        prompt_tokens=result.prompt_tokens,
                        completion_tokens=result.completion_tokens,
                        total_tokens=result.total_tokens
                    )
                )
                
                return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in chat completions: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Chat completion failed: {str(e)}"
            )
    
    # Audio endpoints
    @app.post("/v1/audio/transcriptions")
    async def create_transcription(
        file: UploadFile = File(...),
        model: str = Form("whisper-1"),
        language: Optional[str] = Form(None),
        prompt: Optional[str] = Form(None),
        response_format: str = Form("json"),
        temperature: float = Form(0.0),
        api_key: Optional[str] = Depends(validate_api_key),
        db: Session = Depends(get_db_session)
    ):
        """OpenAI-compatible audio transcription endpoint."""
        try:
            # Validate audio file
            if not file.content_type or not file.content_type.startswith('audio/'):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="File must be an audio file"
                )
            
            # Map OpenAI model names to our audio models
            model_mapping = {
                "whisper-1": "whisper-small-mlx",
                "whisper-large": "whisper-large-mlx",
                "whisper-small": "whisper-small-mlx",
                "whisper-medium": "whisper-medium-mlx",
                "whisper-tiny": "whisper-tiny-mlx",
                "parakeet": "parakeet-tdt-0.6b-v2"
            }
            
            # Get actual model name
            actual_model_name = model_mapping.get(model, model)
            
            # Check if audio model exists and is loaded
            model_record = db.query(Model).filter(Model.name == actual_model_name).first()
            if not model_record:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Audio model '{actual_model_name}' not found. Install it first with POST /v1/models/install"
                )
            
            model_manager = get_model_manager()
            
            # Check if model is loaded, auto-load if not
            loaded_model = model_manager.get_model_for_inference(actual_model_name)
            if not loaded_model:
                logger.info(f"Audio model {actual_model_name} not loaded, attempting to load...")
                success = await model_manager.load_model_async(
                    model_name=actual_model_name,
                    model_path=model_record.path,
                    priority=10  # High priority for audio requests
                )
                if not success:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail=f"Failed to load audio model '{actual_model_name}'"
                    )
                
                # Get the loaded model
                loaded_model = model_manager.get_model_for_inference(actual_model_name)
                if not loaded_model:
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail=f"Audio model '{actual_model_name}' failed to load properly"
                    )
            
            # Check if this is an audio model
            if not hasattr(loaded_model.mlx_wrapper, 'transcribe_audio'):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Model '{actual_model_name}' is not an audio transcription model"
                )
            
            # Save uploaded file temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            try:
                # Transcribe using queued audio processing
                result = await queued_transcribe_audio(
                    model_name=actual_model_name,
                    file_path=temp_file_path,
                    language=language,
                    initial_prompt=prompt,
                    temperature=temperature
                )
                
                # Extract text content from result
                # Note: Usage counting is now handled in the queue manager
                text_content = result.get("text", "")
                
                # Return response based on format
                if response_format == "json":
                    return {"text": text_content}
                elif response_format == "text":
                    return Response(content=text_content, media_type="text/plain")
                elif response_format == "verbose_json":
                    if isinstance(result, dict):
                        return result
                    else:
                        return {"text": text_content}
                else:
                    # For srt, vtt formats - basic implementation
                    return {"text": text_content}
                    
            finally:
                # Clean up temporary file
                os.unlink(temp_file_path)
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in audio transcription: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Transcription failed: {str(e)}"
            )
    
    @app.post("/v1/audio/speech")
    async def create_speech(
        request: AudioSpeechRequest,
        api_key: Optional[str] = Depends(validate_api_key)
    ):
        """OpenAI-compatible text-to-speech endpoint."""
        try:
            # Use MLX Audio for TTS
            try:
                import mlx_audio
            except ImportError:
                raise HTTPException(
                    status_code=status.HTTP_501_NOT_IMPLEMENTED,
                    detail="MLX Audio not installed. Install with: pip install mlx-audio"
                )
            
            # Generate speech
            # Map OpenAI voices to available models
            voice_mapping = {
                "alloy": "kokoro",
                "echo": "kokoro", 
                "fable": "kokoro",
                "onyx": "kokoro",
                "nova": "kokoro",
                "shimmer": "kokoro"
            }
            
            model_name = voice_mapping.get(request.voice, "kokoro")
            
            # Generate audio using queued processing
            audio_content = await queued_generate_speech(
                text=request.input,
                voice=model_name,
                speed=request.speed
            )
            
            # Return audio response
            media_type_mapping = {
                "mp3": "audio/mpeg",
                "opus": "audio/opus", 
                "aac": "audio/aac",
                "flac": "audio/flac",
                "wav": "audio/wav"
            }
            
            media_type = media_type_mapping.get(request.response_format, "audio/wav")
            
            return Response(
                content=audio_content,
                media_type=media_type,
                headers={
                    "Content-Disposition": f"attachment; filename=speech.{request.response_format}"
                }
            )
                
        except ImportError as e:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail=f"Text-to-speech not available: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Error in speech generation: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Speech generation failed: {str(e)}"
            )
    
    @app.get("/v1/models")
    async def list_models_openai_format(
        db: Session = Depends(get_db_session),
        api_key: Optional[str] = Depends(validate_api_key)
    ):
        """OpenAI-compatible models list endpoint."""
        try:
            models = db.query(Model).all()
            
            import time
            
            return {
                "object": "list",
                "data": [
                    {
                        "id": model.name,
                        "object": "model",
                        "created": int(model.created_at.timestamp()) if model.created_at else int(time.time()),
                        "owned_by": "mlx-gui",
                        "permission": [],
                        "root": model.name,
                        "parent": None
                    }
                    for model in models
                ]
            }
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error listing models"
            )
    
    @app.post("/v1/models/install")
    async def install_model(
        request: ModelInstallRequest,
        db: Session = Depends(get_db_session)
    ):
        """Install a model from HuggingFace Hub."""
        try:
            # Get model details from HuggingFace
            hf_client = get_huggingface_client()
            model_info = hf_client.get_model_details(request.model_id)
            
            if not model_info:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model '{request.model_id}' not found on HuggingFace"
                )
            
            if not model_info.mlx_compatible:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Model '{request.model_id}' is not MLX compatible"
                )
            
            # Check system compatibility
            system_monitor = get_system_monitor()
            estimated_memory = model_info.estimated_memory_gb or 4.0  # Default estimate
            can_load, compatibility_message = system_monitor.check_model_compatibility(estimated_memory)
            
            if not can_load:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=compatibility_message
                )
            
            # Use provided name or default to model name
            model_name = request.name or model_info.name
            
            # Check if model already exists
            existing_model = db.query(Model).filter(Model.name == model_name).first()
            if existing_model:
                return {
                    "message": f"Model '{model_name}' already installed",
                    "model_name": model_name,
                    "model_id": request.model_id,
                    "status": "already_installed"
                }
            
            # Create model record in database
            new_model = Model(
                name=model_name,
                path=request.model_id,  # Store HF model ID as path
                version=None,
                model_type=model_info.model_type,
                huggingface_id=request.model_id,
                memory_required_gb=int(estimated_memory),
                status="unloaded"
            )
            
            # Set metadata
            metadata = {
                "author": model_info.author,
                "downloads": model_info.downloads,
                "likes": model_info.likes,
                "tags": model_info.tags,
                "description": model_info.description,
                "size_gb": model_info.size_gb,
                "estimated_memory_gb": model_info.estimated_memory_gb,
                "mlx_repo_id": model_info.mlx_repo_id
            }
            new_model.set_metadata(metadata)
            
            db.add(new_model)
            db.commit()
            
            return {
                "message": f"Model '{model_name}' installed successfully",
                "model_name": model_name,
                "model_id": request.model_id,
                "estimated_memory_gb": estimated_memory,
                "status": "installed"
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error installing model {request.model_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Model installation failed: {str(e)}"
            )
    
    # Model management endpoints
    @app.get("/v1/manager/status")
    async def get_manager_status():
        """Get detailed model manager status."""
        try:
            model_manager = get_model_manager()
            return {
                "loaded_models": model_manager.get_loaded_models(),
                "system_status": model_manager.get_system_status(),
                "queue_status": model_manager._loading_queue.get_queue_status()
            }
        except Exception as e:
            logger.error(f"Error getting manager status: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error getting manager status"
            )
    
    @app.get("/v1/manager/models/{model_name}/status")
    async def get_model_status(model_name: str):
        """Get detailed status of a specific model."""
        try:
            model_manager = get_model_manager()
            status = model_manager.get_model_status(model_name)
            return status
        except Exception as e:
            logger.error(f"Error getting model status for {model_name}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error getting model status"
            )
    
    @app.post("/v1/manager/models/{model_name}/priority")
    async def update_model_priority(model_name: str, priority_data: dict):
        """Update model loading priority in queue."""
        try:
            new_priority = priority_data.get("priority", 0)
            # TODO: Implement priority update in queue
            return {
                "model": model_name,
                "priority": new_priority,
                "message": "Priority update requested"
            }
        except Exception as e:
            logger.error(f"Error updating priority for {model_name}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error updating model priority"
            )
    
    # Admin interface routes
    @app.get("/admin")
    async def admin_interface():
        """Serve the admin interface."""
        from fastapi.responses import HTMLResponse
        from pathlib import Path
        import sys
        
        # Read the admin template - handle both development and bundled app
        if hasattr(sys, 'frozen') and sys.frozen:
            # Running as bundled app - use PyInstaller's _MEIPASS
            if hasattr(sys, '_MEIPASS'):
                template_path = Path(sys._MEIPASS) / "mlx_gui" / "templates" / "admin.html"
                logger.info(f"PyInstaller bundled app: Looking for template at {template_path}")
                logger.info(f"_MEIPASS directory: {sys._MEIPASS}")
                # List contents of _MEIPASS for debugging
                meipass_path = Path(sys._MEIPASS)
                if meipass_path.exists():
                    logger.info(f"_MEIPASS contents: {list(meipass_path.iterdir())}")
                    mlx_gui_path = meipass_path / "mlx_gui"
                    if mlx_gui_path.exists():
                        logger.info(f"mlx_gui directory contents: {list(mlx_gui_path.iterdir())}")
            else:
                # Fallback for frozen apps without _MEIPASS
                template_path = Path(sys.executable).parent / "mlx_gui" / "templates" / "admin.html"
                logger.info(f"Frozen app fallback: Looking for template at {template_path}")
        else:
            # Running in development
            template_path = Path(__file__).parent / "templates" / "admin.html"
            logger.info(f"Development mode: Looking for template at {template_path}")
            
        if not template_path.exists():
            # Try to find template in alternate locations
            logger.error(f"Template not found at {template_path}")
            if hasattr(sys, 'frozen') and sys.frozen:
                # Try some alternate paths in bundled app
                alternate_paths = [
                    Path(sys.executable).parent / "templates" / "admin.html",
                    Path(sys.executable).parent / "admin.html",
                    Path(sys.executable).parent / "Contents" / "Resources" / "templates" / "admin.html",
                ]
                for alt_path in alternate_paths:
                    logger.info(f"Trying alternate path: {alt_path}")
                    if alt_path.exists():
                        template_path = alt_path
                        logger.info(f"Found template at alternate path: {template_path}")
                        break
                else:
                    # List actual directory contents for debugging
                    exec_dir = Path(sys.executable).parent
                    logger.error(f"Executable directory: {exec_dir}")
                    logger.error(f"Executable directory contents: {list(exec_dir.iterdir())}")
            
            if not template_path.exists():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Admin interface template not found at {template_path}"
                )
        
        with open(template_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Replace version placeholder with actual version
        from mlx_gui import __version__
        html_content = html_content.replace('{{ version }}', __version__)
        
        return HTMLResponse(content=html_content)
    
    @app.post("/v1/system/shutdown")
    async def shutdown_server():
        """Gracefully shutdown the server."""
        import asyncio
        import os
        
        def shutdown():
            """Shutdown the server."""
            logger.info("Shutdown requested via API")
            # This will trigger the lifespan cleanup
            os._exit(0)
        
        # Schedule shutdown after a short delay to allow response to be sent
        asyncio.get_event_loop().call_later(1.0, shutdown)
        
        return {"message": "Server shutting down"}
    
    @app.post("/v1/system/restart")
    async def restart_server():
        """Restart the server with updated settings."""
        import asyncio
        import os
        import sys
        import subprocess
        
        def restart():
            """Restart the server."""
            logger.info("Restart requested via API")
            
            # Try to restart using the same command line arguments
            try:
                # Get the current command line
                python_executable = sys.executable
                script_args = sys.argv
                
                # Start new process
                subprocess.Popen([python_executable] + script_args)
                
                # Exit current process
                os._exit(0)
            except Exception as e:
                logger.error(f"Failed to restart: {e}")
                # Fallback to shutdown
                os._exit(0)
        
        # Schedule restart after a short delay to allow response to be sent
        asyncio.get_event_loop().call_later(1.0, restart)
        
        return {"message": "Server restarting with updated settings"}
    
    @app.get("/v1/debug/model/{model_id:path}")
    async def debug_model_info(model_id: str):
        """Debug endpoint to inspect model card content for size estimation troubleshooting."""
        try:
            hf_client = get_huggingface_client()
            
            # Get raw model info
            from huggingface_hub import model_info
            model = model_info(model_id)
            
            # Extract all relevant fields for debugging
            debug_info = {
                "model_id": model_id,
                "description": getattr(model, 'description', None),
                "tags": getattr(model, 'tags', []),
                "library_name": getattr(model, 'library_name', None),
                "pipeline_tag": getattr(model, 'pipeline_tag', None),
                "card_data": getattr(model, 'card_data', None),
                "size_estimation": None,
                "debug_log": []
            }
            
            # Try our size estimation with debug logging
            try:
                model_name = model.id.lower()
                description = getattr(model, 'description', '') or ''
                
                debug_info["debug_log"].append(f"Model name (lower): {model_name}")
                debug_info["debug_log"].append(f"Description length: {len(description)}")
                debug_info["debug_log"].append(f"Description content: {repr(description[:500])}")
                
                # Test our regex patterns
                import re
                param_patterns = [
                    r'(\d+(?:\.\d+)?)\s*[Bb](?:illion)?\s+param',
                    r'(\d+(?:\.\d+)?)\s*[Bb](?:illion)?\s+parameter', 
                    r'(\d+(?:\.\d+)?)\s*[Bb](?:illion)?\s+model',
                    r'(\d+(?:\.\d+)?)\s*[Bb](?:illion)?\s+weights',
                    r'(\d+(?:\.\d+)?)\s*[Bb](?:illion)?\s*-?\s*param',
                    r'Parameters?:\s*(\d+(?:\.\d+)?)\s*[Bb]',
                    r'Model size:\s*(\d+(?:\.\d+)?)\s*[Bb]',
                    r'(\d+(?:\.\d+)?)\s*[Bb](?:illion)?\s*parameter',
                ]
                
                param_count_billions = None
                for i, pattern in enumerate(param_patterns):
                    matches = re.findall(pattern, description, re.IGNORECASE)
                    debug_info["debug_log"].append(f"Pattern {i+1} '{pattern}': {matches}")
                    if matches:
                        param_count_billions = float(matches[0])
                        debug_info["debug_log"].append(f"Found parameter count: {param_count_billions}B")
                        break
                
                # Test quantization detection
                quantization_info = {
                    "detected_bits": 16,  # default
                    "indicators": []
                }
                
                if "4bit" in model_name or "4-bit" in model_name:
                    quantization_info["detected_bits"] = 4
                    quantization_info["indicators"].append("4bit/4-bit")
                elif "8bit" in model_name or "8-bit" in model_name:
                    quantization_info["detected_bits"] = 8
                    quantization_info["indicators"].append("8bit/8-bit")
                
                debug_info["quantization"] = quantization_info
                
                if param_count_billions:
                    bits_per_param = quantization_info["detected_bits"]
                    base_memory_gb = (param_count_billions * 1e9 * bits_per_param) / (8 * 1024**3)
                    total_memory_gb = base_memory_gb * 1.25
                    
                    debug_info["size_estimation"] = {
                        "param_count_billions": param_count_billions,
                        "bits_per_param": bits_per_param,
                        "base_memory_gb": base_memory_gb,
                        "total_memory_gb": total_memory_gb,
                        "calculation": f"{param_count_billions}B × {bits_per_param}bit ÷ 8 ÷ 1024³ × 1.25 = {total_memory_gb:.2f}GB"
                    }
                else:
                    debug_info["debug_log"].append("No parameter count found in description")
                
                # Also check safetensors metadata 
                debug_info["debug_log"].append("Checking safetensors metadata...")
                try:
                    if hasattr(model, 'safetensors') and model.safetensors:
                        debug_info["debug_log"].append(f"Found safetensors metadata: {model.safetensors}")
                        for file_name, metadata in model.safetensors.items():
                            debug_info["debug_log"].append(f"File {file_name}: {metadata}")
                            if isinstance(metadata, dict):
                                if 'total' in metadata:
                                    total_params = metadata.get('total', 0)
                                    debug_info["debug_log"].append(f"Found 'total' in metadata: {total_params}")
                                    if total_params > 1000000:
                                        params_in_billions = total_params / 1e9
                                        debug_info["debug_log"].append(f"Calculated parameter count: {params_in_billions}B")
                    else:
                        debug_info["debug_log"].append("No safetensors metadata found")
                except Exception as e:
                    debug_info["debug_log"].append(f"Error checking safetensors: {str(e)}")
                
            except Exception as e:
                debug_info["debug_log"].append(f"Error in size estimation: {str(e)}")
            
            return debug_info
            
        except Exception as e:
            logger.error(f"Error debugging model {model_id}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error debugging model: {str(e)}"
            )
    
    return app