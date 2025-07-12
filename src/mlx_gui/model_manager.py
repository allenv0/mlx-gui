"""
Model management and loading system for MLX-GUI.
Handles model lifecycle, queue management, and MLX-LM integration.
"""

import asyncio
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Any, List, Callable, AsyncGenerator, Tuple
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import lru_cache

import mlx.core as mx
from sqlalchemy.orm import Session

from mlx_gui.database import get_database_manager
from mlx_gui.models import Model, ModelStatus, InferenceRequest, RequestQueue, QueueStatus
from mlx_gui.system_monitor import get_system_monitor
from mlx_gui.mlx_integration import get_inference_engine, MLXModelWrapper, GenerationConfig, GenerationResult

logger = logging.getLogger(__name__)

# Global cache for model memory calculations
_model_memory_cache: Dict[str, float] = {}
_model_memory_cache_lock = threading.RLock()
_model_memory_cache_timestamp = 0
_model_memory_cache_ttl = 300  # 5 minutes TTL


class LoadingStatus(Enum):
    """Model loading status."""
    IDLE = "idle"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"
    FAILED = "failed"


@dataclass
class LoadedModel:
    """Container for a loaded MLX model."""
    model_id: str
    mlx_wrapper: MLXModelWrapper  # The MLX model wrapper
    loaded_at: datetime
    last_used_at: datetime
    memory_usage_gb: float
    
    def update_last_used(self):
        """Update the last used timestamp."""
        self.last_used_at = datetime.utcnow()
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.mlx_wrapper.config


@dataclass
class LoadRequest:
    """Model loading request."""
    model_name: str
    model_path: str
    priority: int = 0
    requester_id: str = "system"
    callback: Optional[Callable] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class ModelLoadingQueue:
    """Thread-safe queue for model loading requests."""
    
    def __init__(self):
        self._queue: List[LoadRequest] = []
        self._lock = threading.Lock()
        self._event = threading.Event()
    
    def add_request(self, request: LoadRequest) -> int:
        """Add a loading request to the queue."""
        with self._lock:
            self._queue.append(request)
            # Sort by priority (higher first), then by creation time
            self._queue.sort(key=lambda x: (-x.priority, x.created_at))
            position = self._queue.index(request)
            self._event.set()
            return position
    
    def get_next_request(self, timeout: Optional[float] = None) -> Optional[LoadRequest]:
        """Get the next request from the queue."""
        if timeout:
            self._event.wait(timeout)
        
        with self._lock:
            if self._queue:
                request = self._queue.pop(0)
                if not self._queue:
                    self._event.clear()
                return request
            return None
    
    def remove_request(self, model_name: str) -> bool:
        """Remove a request from the queue."""
        with self._lock:
            for i, request in enumerate(self._queue):
                if request.model_name == model_name:
                    self._queue.pop(i)
                    return True
            return False
    
    def get_queue_status(self) -> List[Dict[str, Any]]:
        """Get current queue status."""
        with self._lock:
            return [
                {
                    "model_name": req.model_name,
                    "priority": req.priority,
                    "requester_id": req.requester_id,
                    "created_at": req.created_at.isoformat(),
                    "position": i
                }
                for i, req in enumerate(self._queue)
            ]
    
    def size(self) -> int:
        """Get queue size."""
        with self._lock:
            return len(self._queue)


class ModelManager:
    """
    Central model management system.
    Handles loading, unloading, and lifecycle management of MLX models.
    """
    
    def __init__(self, max_concurrent_models: int = 3, max_memory_usage: float = 0.8):
        self.max_memory_usage = max_memory_usage  # Max % of system memory to use
        
        # Read max_concurrent_models from database settings
        self.max_concurrent_models = self._get_max_concurrent_models(max_concurrent_models)
        
        # Model storage
        self._loaded_models: Dict[str, LoadedModel] = {}
        self._loading_status: Dict[str, LoadingStatus] = {}
        self._lock = threading.RLock()
        
        # Queue system
        self._loading_queue = ModelLoadingQueue()
        self._queue_worker_thread = None
        self._cleanup_worker_thread = None
        self._queue_worker_running = False
        self._shutdown_requested = False
        
        # Thread pool for non-blocking operations
        self._thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="model_manager")
        
        # Register cleanup on exit
        import atexit
        atexit.register(self._force_cleanup)
        
        # Auto-unload settings - read from database with 5 minute default
        self._auto_unload_enabled = True
        self._inactivity_timeout = self._get_inactivity_timeout()
        self._cleanup_interval = 60  # seconds
        
        # Performance optimizations
        self._last_cleanup_time = time.time()
        self._model_memory_cache = {}
        
        # Don't start background workers immediately - start them lazily
    
    def _get_max_concurrent_models(self, default_value: int) -> int:
        """Get the max concurrent models from database settings."""
        try:
            db_manager = get_database_manager()
            return db_manager.get_setting("max_concurrent_models", default_value)
        except Exception as e:
            logger.warning(f"Failed to read max concurrent models from database: {e}")
            return default_value

    def _get_inactivity_timeout(self) -> timedelta:
        """Get the inactivity timeout from database settings, defaulting to 5 minutes."""
        try:
            db_manager = get_database_manager()
            timeout_minutes = db_manager.get_setting("model_inactivity_timeout_minutes", 5)
            return timedelta(minutes=timeout_minutes)
        except Exception as e:
            logger.warning(f"Failed to read inactivity timeout from database: {e}")
            return timedelta(minutes=5)  # Default to 5 minutes
    
    def _start_queue_worker(self):
        """Start the queue processing worker."""
        if not self._queue_worker_running:
            self._queue_worker_running = True
            # Create daemon thread that won't prevent shutdown
            self._queue_worker_thread = threading.Thread(
                target=self._queue_worker,
                name="model_loader_queue",
                daemon=True
            )
            self._queue_worker_thread.start()
            logger.info("Model loading queue worker started")
    
    def _start_cleanup_worker(self):
        """Start the cleanup worker."""
        if not hasattr(self, '_cleanup_worker_running') or not self._cleanup_worker_running:
            self._cleanup_worker_running = True
            self._cleanup_worker_thread = threading.Thread(
                target=self._cleanup_worker,
                name="model_cleanup",
                daemon=True
            )
            self._cleanup_worker_thread.start()
            logger.info("Model cleanup worker started")
    
    def _queue_worker(self):
        """Background worker that processes the loading queue."""
        logger.info("Model loading queue worker started")
        
        while not self._shutdown_requested:
            try:
                # Get next request with timeout
                request = self._loading_queue.get_next_request(timeout=1.0)
                
                if request:
                    # Process in thread pool to avoid blocking
                    self._thread_pool.submit(self._load_model_sync, request)
                else:
                    # No requests, sleep briefly
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error in queue worker: {e}")
                time.sleep(1.0)  # Wait longer on errors
        
        logger.info("Model loading queue worker stopped")
    
    def _cleanup_worker(self):
        """Background worker that cleans up inactive models."""
        logger.info("Model cleanup worker started")
        
        while not self._shutdown_requested:
            try:
                current_time = time.time()
                
                # Only run cleanup every cleanup_interval seconds
                if current_time - self._last_cleanup_time >= self._cleanup_interval:
                    self._cleanup_inactive_models()
                    self._last_cleanup_time = current_time
                
                # Sleep for a shorter interval
                time.sleep(10.0)  # Check every 10 seconds instead of 60
                
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")
                time.sleep(30.0)  # Wait longer on errors
        
        logger.info("Model cleanup worker stopped")
    
    def shutdown(self):
        """Shutdown the model manager."""
        self._shutdown_requested = True
        self._thread_pool.shutdown(wait=True)
        logger.info("Model manager shutdown complete")
    
    def _force_cleanup(self):
        """Force cleanup on exit."""
        try:
            with self._lock:
                for model_name in list(self._loaded_models.keys()):
                    self._unload_model_internal(model_name)
        except Exception as e:
            logger.error(f"Error during force cleanup: {e}")
    
    def _cleanup_inactive_models(self):
        """Clean up models that have been inactive for too long."""
        if not self._auto_unload_enabled:
            return
        
        current_time = datetime.utcnow()
        cutoff_time = current_time - self._inactivity_timeout
        
        with self._lock:
            models_to_unload = []
            for model_name, loaded_model in self._loaded_models.items():
                if loaded_model.last_used_at < cutoff_time:
                    models_to_unload.append(model_name)
            
            # Unload inactive models
            for model_name in models_to_unload:
                logger.info(f"Auto-unloading inactive model: {model_name}")
                self._unload_model_internal(model_name)
    
    def _set_loading_status(self, model_name: str, status: LoadingStatus):
        """Set the loading status for a model."""
        self._loading_status[model_name] = status
        
        # Update database status
        try:
            db_manager = get_database_manager()
            with db_manager.get_session() as session:
                model_record = session.query(Model).filter(Model.name == model_name).first()
                if model_record:
                    if status == LoadingStatus.LOADED:
                        model_record.status = ModelStatus.LOADED.value
                    elif status == LoadingStatus.LOADING:
                        model_record.status = ModelStatus.LOADING.value
                    elif status == LoadingStatus.FAILED:
                        model_record.status = ModelStatus.FAILED.value
                    elif status == LoadingStatus.IDLE:
                        model_record.status = ModelStatus.UNLOADED.value
                    session.commit()
        except Exception as e:
            logger.error(f"Error updating model status in database: {e}")
    
    @lru_cache(maxsize=128)
    def _calculate_actual_memory_usage(self, model_path: str) -> float:
        """Calculate the actual memory usage based on file sizes + overhead with caching."""
        global _model_memory_cache, _model_memory_cache_timestamp
        
        current_time = time.time()
        
        # Check cache first
        with _model_memory_cache_lock:
            if current_time - _model_memory_cache_timestamp < _model_memory_cache_ttl:
                if model_path in _model_memory_cache:
                    return _model_memory_cache[model_path]
            else:
                # Cache expired, clear it
                _model_memory_cache.clear()
                _model_memory_cache_timestamp = current_time
        
        total_size_bytes = 0
        
        try:
            # Handle both local paths and HuggingFace model IDs
            actual_path = self._resolve_model_path(model_path)
            
            # Walk through all files in the model directory
            for root, dirs, files in os.walk(actual_path):
                for file in files:
                    # Count all model-related files
                    if file.endswith(('.safetensors', '.bin', '.pth', '.pt', '.gguf', '.npz')):
                        file_path = os.path.join(root, file)
                        if os.path.exists(file_path):
                            total_size_bytes += os.path.getsize(file_path)
            
            # Convert to GB
            file_size_gb = total_size_bytes / (1024**3)
            
            # Add MLX overhead (25% for inference, activations, etc.)
            # Audio models might need less overhead, text models might need more
            if "whisper" in model_path.lower() or "parakeet" in model_path.lower():
                overhead_multiplier = 1.15  # 15% overhead for audio models
            else:
                overhead_multiplier = 1.25  # 25% overhead for text models
                
            actual_memory_gb = file_size_gb * overhead_multiplier
            
            # Round to one decimal place
            actual_memory_gb = round(actual_memory_gb, 1)
            
            # Cache the result
            with _model_memory_cache_lock:
                _model_memory_cache[model_path] = max(actual_memory_gb, 0.1)
            
            logger.info(f"Model {model_path} -> {actual_path} file size: {file_size_gb:.1f}GB, "
                       f"with overhead: {actual_memory_gb:.1f}GB "
                       f"(multiplier: {overhead_multiplier})")
            
            return max(actual_memory_gb, 0.1)  # Minimum 0.1GB
            
        except Exception as e:
            logger.warning(f"Could not calculate actual memory usage for {model_path}: {e}")
            # Fallback to a reasonable default
            return 2.0
    
    @lru_cache(maxsize=128)
    def _resolve_model_path(self, model_path: str) -> str:
        """Resolve model path with caching for better performance."""
        if model_path.startswith("~"):
            return os.path.expanduser(model_path)
        
        # Handle HuggingFace cache paths
        if "huggingface" in model_path.lower() or "/.cache/huggingface/" in model_path:
            # This is already a resolved path
            return model_path
        
        # Check if it's a HuggingFace model ID
        if "/" in model_path and not os.path.exists(model_path):
            try:
                from huggingface_hub import snapshot_download
                return snapshot_download(repo_id=model_path, local_files_only=True)
            except Exception:
                pass
        
        return model_path
    
    def _check_memory_constraints(self, required_memory_gb: float) -> Tuple[bool, str]:
        """Check if we can load a model with the required memory. Returns (can_load, warning_message)."""
        # First check concurrent model count limit - this is still a hard limit
        if len(self._loaded_models) >= self.max_concurrent_models:
            return False, f"Maximum concurrent models ({self.max_concurrent_models}) already loaded"
        
        system_monitor = get_system_monitor()
        memory_info = system_monitor.get_memory_info()
        
        # Calculate current usage by loaded models
        current_model_memory = sum(m.memory_usage_gb for m in self._loaded_models.values())
        
        # Check if we have enough total memory (80% of system RAM)
        max_allowed_memory = memory_info.total_gb * 0.8
        if current_model_memory + required_memory_gb > max_allowed_memory:
            warning_msg = f"⚠️ Memory warning: Loading this model ({required_memory_gb:.1f}GB) with current models ({current_model_memory:.1f}GB) may exceed recommended memory ({max_allowed_memory:.1f}GB of {memory_info.total_gb:.1f}GB total)"
            return True, warning_msg  # Allow but warn
            
        return True, ""
    
    def _load_model_sync(self, request: LoadRequest):
        """Synchronously load a model (runs in background thread)."""
        model_name = request.model_name
        model_path = request.model_path
        
        try:
            self._set_loading_status(model_name, LoadingStatus.LOADING)
            
            # Check if already loaded
            with self._lock:
                if model_name in self._loaded_models:
                    logger.info(f"Model {model_name} already loaded")
                    self._loaded_models[model_name].update_last_used()
                    return
            
            # Get model info from database
            db_manager = get_database_manager()
            with db_manager.get_session() as session:
                model_record = session.query(Model).filter(Model.name == model_name).first()
                if not model_record:
                    raise ValueError(f"Model {model_name} not found in database")
                
                # Check memory constraints
                can_load, memory_warning = self._check_memory_constraints(model_record.memory_required_gb)
                if not can_load:
                    # Hard limits still apply (e.g., max concurrent models)
                    raise RuntimeError(memory_warning)
                elif memory_warning:
                    # Log warning but proceed
                    logger.warning(f"Loading model {model_name} with memory warning: {memory_warning}")
                
                # Also check system compatibility for hardware requirements
                system_monitor = get_system_monitor()
                hardware_compatible, hardware_message = system_monitor.check_model_compatibility(model_record.memory_required_gb)
                if not hardware_compatible:
                    # Hardware compatibility is still a hard requirement
                    raise RuntimeError(hardware_message)
                elif "warning" in hardware_message.lower():
                    # Log system memory warning
                    logger.warning(f"Loading model {model_name} with system warning: {hardware_message}")
            
            # Free up space if needed
            self._ensure_capacity_for_model(model_record.memory_required_gb)
            
            # Load the model using MLX-LM
            logger.info(f"Loading MLX model from {model_path}")
            
            inference_engine = get_inference_engine()
            mlx_wrapper = inference_engine.load_model(model_name, model_path)
            
            # Calculate actual file size + overhead for this model
            actual_memory = self._calculate_actual_memory_usage(model_path)
            
            # Update database with actual memory usage
            db_manager = get_database_manager()
            with db_manager.get_session() as session:
                model_record = session.query(Model).filter(Model.name == model_name).first()
                if model_record:
                    model_record.memory_required_gb = actual_memory
                    session.commit()
                    logger.info(f"Updated {model_name} memory requirement in DB: {actual_memory:.1f}GB")
            
            loaded_model = LoadedModel(
                model_id=model_name,
                mlx_wrapper=mlx_wrapper,
                loaded_at=datetime.utcnow(),
                last_used_at=datetime.utcnow(),
                memory_usage_gb=actual_memory
            )
            
            with self._lock:
                self._loaded_models[model_name] = loaded_model
            
            self._set_loading_status(model_name, LoadingStatus.LOADED)
            logger.info(f"Successfully loaded model {model_name}")
            
            # Call callback if provided
            if request.callback:
                try:
                    request.callback(model_name, True, None)
                except Exception as e:
                    logger.error(f"Error in model load callback: {e}")
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            self._set_loading_status(model_name, LoadingStatus.FAILED)
            
            # Call callback with error if provided
            if request.callback:
                try:
                    request.callback(model_name, False, str(e))
                except Exception as callback_error:
                    logger.error(f"Error in model load error callback: {callback_error}")
    
    def _ensure_capacity_for_model(self, required_memory_gb: float):
        """Ensure we have capacity for a new model by unloading others if needed."""
        system_monitor = get_system_monitor()
        memory_info = system_monitor.get_memory_info()
        
        # Calculate current usage
        current_model_memory = sum(m.memory_usage_gb for m in self._loaded_models.values())
        max_allowed_memory = memory_info.total_gb * 0.8
        
        # If we need more memory, unload least recently used models
        while current_model_memory + required_memory_gb > max_allowed_memory and self._loaded_models:
            # Find least recently used model
            lru_model = min(self._loaded_models.values(), key=lambda m: m.last_used_at)
            lru_name = lru_model.model_id
            
            logger.info(f"Unloading {lru_name} to make room for new model")
            self._unload_model_internal(lru_name)
            current_model_memory -= lru_model.memory_usage_gb
    
    def _unload_model_internal(self, model_name: str):
        """Internal method to unload a model."""
        try:
            with self._lock:
                if model_name not in self._loaded_models:
                    return False
                
                loaded_model = self._loaded_models.pop(model_name)
            
            # Unload from inference engine
            inference_engine = get_inference_engine()
            inference_engine.unload_model(model_name)
            
            # Update database
            db_manager = get_database_manager()
            with db_manager.get_session() as session:
                model_record = session.query(Model).filter(Model.name == model_name).first()
                if model_record:
                    model_record.status = ModelStatus.UNLOADED.value
                    model_record.last_unloaded_at = datetime.utcnow()
                    session.commit()
            
            self._set_loading_status(model_name, LoadingStatus.IDLE)
            logger.info(f"Successfully unloaded model {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error unloading model {model_name}: {e}")
            return False
    
    async def load_model_async(self, model_name: str, model_path: str, priority: int = 0) -> bool:
        """Load a model asynchronously."""
        # Start background workers if not already running
        self._start_queue_worker()
        self._start_cleanup_worker()
        
        # Check if already loaded
        with self._lock:
            if model_name in self._loaded_models:
                self._loaded_models[model_name].update_last_used()
                return True
        
        # Check if already loading
        if self._loading_status.get(model_name) == LoadingStatus.LOADING:
            logger.info(f"Model {model_name} is already being loaded")
            return True
        
        # Create completion future
        completion_future = asyncio.Future()
        
        def load_callback(model_name: str, success: bool, error: Optional[str]):
            if success:
                completion_future.set_result(True)
            else:
                completion_future.set_exception(RuntimeError(error or "Unknown error"))
        
        # Create load request
        request = LoadRequest(
            model_name=model_name,
            model_path=model_path,
            priority=priority,
            callback=load_callback
        )
        
        # Add to queue
        position = self._loading_queue.add_request(request)
        logger.info(f"Queued model {model_name} for loading (position: {position})")
        
        # Wait for completion with timeout
        try:
            await asyncio.wait_for(completion_future, timeout=300.0)  # 5 minute timeout
            return True
        except asyncio.TimeoutError:
            raise RuntimeError(f"Model loading timed out after 5 minutes: {model_name}")
    
    def unload_model(self, model_name: str) -> bool:
        """Unload a model."""
        return self._unload_model_internal(model_name)
    
    def get_loaded_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about loaded models."""
        with self._lock:
            return {
                name: {
                    "model_id": model.model_id,
                    "loaded_at": model.loaded_at.isoformat(),
                    "last_used_at": model.last_used_at.isoformat(),
                    "memory_usage_gb": model.memory_usage_gb,
                    "config": model.config
                }
                for name, model in self._loaded_models.items()
            }
    
    def get_model_status(self, model_name: str) -> Dict[str, Any]:
        """Get status of a specific model."""
        with self._lock:
            if model_name in self._loaded_models:
                model = self._loaded_models[model_name]
                return {
                    "status": "loaded",
                    "loaded_at": model.loaded_at.isoformat(),
                    "last_used_at": model.last_used_at.isoformat(),
                    "memory_usage_gb": model.memory_usage_gb
                }
            elif model_name in self._loading_status:
                return {"status": self._loading_status[model_name].value}
            else:
                return {"status": "unloaded"}
    
    def refresh_settings(self):
        """Refresh settings from database."""
        self.max_concurrent_models = self._get_max_concurrent_models(3)
        self._inactivity_timeout = self._get_inactivity_timeout()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status information."""
        system_monitor = get_system_monitor()
        memory_info = system_monitor.get_memory_info()
        
        with self._lock:
            loaded_models_count = len(self._loaded_models)
            total_model_memory = sum(m.memory_usage_gb for m in self._loaded_models.values())
            queue_size = self._loading_queue.size()
        
        return {
            "memory": {
                "total_gb": memory_info.total_gb,
                "available_gb": memory_info.available_gb,
                "used_gb": memory_info.used_gb,
                "percent_used": memory_info.percent_used
            },
            "models": {
                "loaded_count": loaded_models_count,
                "max_concurrent": self.max_concurrent_models,
                "total_memory_gb": total_model_memory,
                "queue_size": queue_size
            },
            "settings": {
                "auto_unload_enabled": self._auto_unload_enabled,
                "inactivity_timeout_minutes": self._inactivity_timeout.total_seconds() / 60
            }
        }
    
    async def generate_text(self, model_name: str, prompt: str, config: GenerationConfig) -> GenerationResult:
        """Generate text using a loaded model."""
        # Ensure model is loaded
        with self._lock:
            if model_name not in self._loaded_models:
                raise ValueError(f"Model {model_name} is not loaded")
            
            # Update last used time
            self._loaded_models[model_name].update_last_used()
        
        # Perform inference
        inference_engine = get_inference_engine()
        result = inference_engine.generate(model_name, prompt, config)
        
        # Update model usage in database
        try:
            db_manager = get_database_manager()
            with db_manager.get_session() as session:
                model_record = session.query(Model).filter(Model.name == model_name).first()
                if model_record:
                    model_record.increment_use_count()
                    session.commit()
        except Exception as e:
            logger.error(f"Error updating model usage: {e}")
        
        return result
    
    async def generate_text_stream(self, model_name: str, prompt: str, config: GenerationConfig) -> AsyncGenerator[str, None]:
        """Generate streaming text using a loaded model."""
        # Ensure model is loaded
        with self._lock:
            if model_name not in self._loaded_models:
                raise ValueError(f"Model {model_name} is not loaded")
            
            # Update last used time
            self._loaded_models[model_name].update_last_used()
        
        # Perform streaming inference
        inference_engine = get_inference_engine()
        async for chunk in inference_engine.generate_stream(model_name, prompt, config):
            yield chunk
        
        # Update model usage in database
        try:
            db_manager = get_database_manager()
            with db_manager.get_session() as session:
                model_record = session.query(Model).filter(Model.name == model_name).first()
                if model_record:
                    model_record.increment_use_count()
                    session.commit()
        except Exception as e:
            logger.error(f"Error updating model usage: {e}")
    
    def get_model_for_inference(self, model_name: str) -> Optional[LoadedModel]:
        """Get a loaded model for inference."""
        with self._lock:
            return self._loaded_models.get(model_name)
    
    def shutdown(self):
        """Shutdown the model manager."""
        self._shutdown_requested = True
        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)
        logger.info("Model manager shutdown complete")


# Global model manager instance
_model_manager: Optional[ModelManager] = None
_model_manager_lock = threading.Lock()


def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    global _model_manager
    
    if _model_manager is None:
        with _model_manager_lock:
            if _model_manager is None:
                _model_manager = ModelManager()
    
    return _model_manager


def shutdown_model_manager():
    """Shutdown the global model manager."""
    global _model_manager
    if _model_manager:
        _model_manager.shutdown()
        _model_manager = None