"""
Inference Request Queue Manager

Handles queuing and processing of inference requests to prevent
concurrency issues and provide better user experience for 2-5 concurrent users.
"""

import asyncio
import json
import logging
import threading
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from functools import lru_cache
import time

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc

from .database import get_database_manager
from .models import Model, RequestQueue, InferenceRequest, QueueStatus, InferenceStatus
from .mlx_integration import GenerationConfig

logger = logging.getLogger(__name__)

# Global cache for queue status
_queue_status_cache: Dict[str, Dict[str, Any]] = {}
_queue_status_cache_lock = threading.RLock()
_queue_status_cache_timestamp = 0
_queue_status_cache_ttl = 5  # 5 seconds TTL


@dataclass
class QueuedRequest:
    """Represents a queued inference request."""
    session_id: str
    model_name: str
    prompt: str
    config: GenerationConfig
    priority: int = 0
    created_at: datetime = None
    callback: Optional[Callable] = None
    streaming: bool = False
    stream_callback: Optional[Callable] = None  # For streaming responses
    request_type: str = "text"  # "text", "transcription", "tts", "embeddings", "vision_generation"
    audio_data: Optional[Dict] = None  # For audio requests
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class InferenceRequestManager:
    """
    Manages queuing and processing of inference requests.
    
    Features:
    - Per-model concurrency limits
    - FIFO queuing with priority support
    - Request serialization
    - Background processing
    - Performance optimizations with caching
    """
    
    def __init__(self, max_concurrent_per_model: int = 1):
        self.max_concurrent_per_model = max_concurrent_per_model
        
        # Track active requests per model
        self._active_requests: Dict[str, int] = {}
        self._lock = threading.RLock()
        
        # Background worker
        self._worker_running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._shutdown_requested = False
        
        # Request tracking for callbacks
        self._pending_callbacks: Dict[str, Callable] = {}
        
        # Performance optimizations
        self._model_id_cache: Dict[str, int] = {}
        self._last_queue_processing = time.time()
        self._processing_interval = 0.5  # Process queue every 500ms instead of 2s
        
    def start(self):
        """Start the background queue processor."""
        if not self._worker_running:
            self._worker_running = True
            self._shutdown_requested = False
            self._worker_thread = threading.Thread(
                target=self._queue_worker,
                name="inference_queue_worker",
                daemon=True
            )
            self._worker_thread.start()
            logger.info("Inference queue worker started")
    
    def stop(self):
        """Stop the background queue processor."""
        self._shutdown_requested = True
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
        self._worker_running = False
        logger.info("Inference queue worker stopped")
    
    @lru_cache(maxsize=128)
    def _get_model_id(self, model_name: str) -> Optional[int]:
        """Get model ID with caching for better performance."""
        try:
            db_manager = get_database_manager()
            with db_manager.get_session() as session:
                model_record = session.query(Model.id).filter(Model.name == model_name).first()
                return model_record.id if model_record else None
        except Exception as e:
            logger.error(f"Error getting model ID for {model_name}: {e}")
            return None
    
    async def queue_request(self, request: QueuedRequest) -> str:
        """
        Queue an inference request.
        
        Returns:
            str: Request ID for tracking
        """
        request_id = str(uuid.uuid4())
        
        with self._lock:
            # Check if we can process immediately
            active_count = self._active_requests.get(request.model_name, 0)
            can_process_now = active_count < self.max_concurrent_per_model
            
            # Get model ID with caching
            model_id = self._get_model_id(request.model_name)
            if not model_id:
                raise ValueError(f"Model {request.model_name} not found in database")
            
            # Always queue the request for persistence and tracking
            db_manager = get_database_manager()
            with db_manager.get_session() as session:
                # Create queue entry
                queue_item = RequestQueue(
                    session_id=request.session_id,
                    model_id=model_id,
                    priority=request.priority,
                    status=QueueStatus.QUEUED.value if not can_process_now else QueueStatus.PROCESSING.value
                )
                
                # Store request data
                request_data = {
                    "request_id": request_id,
                    "prompt": request.prompt,
                    "streaming": request.streaming,
                    "request_type": request.request_type,
                    "audio_data": request.audio_data,
                    "config": {
                        "max_tokens": request.config.max_tokens,
                        "temperature": request.config.temperature,
                        "top_p": request.config.top_p,
                        "top_k": request.config.top_k,
                        "repetition_penalty": request.config.repetition_penalty,
                        "seed": request.config.seed
                    }
                }
                queue_item.set_request_data(request_data)
                
                session.add(queue_item)
                session.commit()
                
                if can_process_now:
                    # Mark as processing immediately
                    queue_item.start_processing()
                    self._active_requests[request.model_name] = active_count + 1
                    session.commit()
                    logger.info(f"Processing request {request_id} immediately for model {request.model_name}")
                else:
                    logger.info(f"Queued request {request_id} for model {request.model_name} (active: {active_count})")
                
                # Store callback if provided
                if request.callback:
                    self._pending_callbacks[request_id] = request.callback
                elif request.stream_callback:
                    self._pending_callbacks[request_id] = request.stream_callback
        
        return request_id
    
    def get_queue_status(self, model_name: str) -> Dict[str, Any]:
        """Get queue status for a specific model with caching."""
        global _queue_status_cache, _queue_status_cache_timestamp
        
        current_time = time.time()
        
        # Check cache first
        with _queue_status_cache_lock:
            if current_time - _queue_status_cache_timestamp < _queue_status_cache_ttl:
                if model_name in _queue_status_cache:
                    cached_status = _queue_status_cache[model_name].copy()
                    # Add real-time active count
                    with self._lock:
                        cached_status["active_requests"] = self._active_requests.get(model_name, 0)
                        cached_status["can_accept_immediate"] = cached_status["active_requests"] < self.max_concurrent_per_model
                    return cached_status
            else:
                # Cache expired, clear it
                _queue_status_cache.clear()
                _queue_status_cache_timestamp = current_time
        
        # Fetch from database
        db_manager = get_database_manager()
        with db_manager.get_session() as session:
            model_record = session.query(Model).filter(Model.name == model_name).first()
            if not model_record:
                return {"error": "Model not found"}
            
            # Count queued requests
            queued_count = session.query(RequestQueue).filter(
                and_(
                    RequestQueue.model_id == model_record.id,
                    RequestQueue.status == QueueStatus.QUEUED.value
                )
            ).count()
            
            # Count processing requests
            processing_count = session.query(RequestQueue).filter(
                and_(
                    RequestQueue.model_id == model_record.id,
                    RequestQueue.status == QueueStatus.PROCESSING.value
                )
            ).count()
            
            with self._lock:
                active_count = self._active_requests.get(model_name, 0)
            
            status = {
                "model_name": model_name,
                "queued_requests": queued_count,
                "processing_requests": processing_count,
                "active_requests": active_count,
                "max_concurrent": self.max_concurrent_per_model,
                "can_accept_immediate": active_count < self.max_concurrent_per_model
            }
            
            # Cache the result
            with _queue_status_cache_lock:
                _queue_status_cache[model_name] = status.copy()
            
            return status
    
    def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific request."""
        db_manager = get_database_manager()
        with db_manager.get_session() as session:
            # First check inference requests table
            inference_request = session.query(InferenceRequest).filter(
                InferenceRequest.request_id == request_id
            ).first()
            
            if inference_request:
                return {
                    "request_id": request_id,
                    "status": inference_request.status,
                    "created_at": inference_request.created_at.isoformat() if inference_request.created_at else None,
                    "completed_at": inference_request.completed_at.isoformat() if inference_request.completed_at else None,
                    "duration_ms": inference_request.duration_ms,
                    "error_message": inference_request.error_message
                }
            
            # Check queue table
            queue_item = session.query(RequestQueue).join(Model).filter(
                RequestQueue.request_data.contains(request_id)
            ).first()
            
            if queue_item:
                request_data = queue_item.get_request_data()
                return {
                    "request_id": request_id,
                    "status": queue_item.status,
                    "created_at": queue_item.created_at.isoformat() if queue_item.created_at else None,
                    "started_at": queue_item.started_at.isoformat() if queue_item.started_at else None,
                    "queue_position": self._get_queue_position(queue_item) if queue_item.status == QueueStatus.QUEUED.value else None
                }
            
            return None
    
    def _get_queue_position(self, queue_item: RequestQueue) -> int:
        """Get position of a queue item in the queue."""
        db_manager = get_database_manager()
        with db_manager.get_session() as session:
            position = session.query(RequestQueue).filter(
                and_(
                    RequestQueue.model_id == queue_item.model_id,
                    RequestQueue.status == QueueStatus.QUEUED.value,
                    or_(
                        RequestQueue.priority > queue_item.priority,
                        and_(
                            RequestQueue.priority == queue_item.priority,
                            RequestQueue.created_at < queue_item.created_at
                        )
                    )
                )
            ).count()
            
            return position + 1
    
    def _queue_worker(self):
        """Background worker that processes queued requests."""
        logger.info("Inference queue worker started")
        
        while not self._shutdown_requested:
            try:
                self._process_next_requests()
                # Check every 500ms instead of 2s for better responsiveness
                threading.Event().wait(self._processing_interval)
                
            except Exception as e:
                logger.error(f"Error in queue worker: {e}")
                threading.Event().wait(2.0)  # Wait longer on errors
        
        logger.info("Inference queue worker stopped")
    
    def _process_next_requests(self):
        """Process next available requests from all model queues."""
        current_time = time.time()
        
        # Throttle processing to avoid excessive database queries
        if current_time - self._last_queue_processing < 0.1:  # Max 10 queries per second
            return
        
        self._last_queue_processing = current_time
        
        db_manager = get_database_manager()
        
        with db_manager.get_session() as session:
            # Get all models with queued requests - use more efficient query
            models_with_queue = session.query(Model.id, Model.name).join(RequestQueue).filter(
                RequestQueue.status == QueueStatus.QUEUED.value
            ).distinct().all()
            
            for model_id, model_name in models_with_queue:
                with self._lock:
                    active_count = self._active_requests.get(model_name, 0)
                    can_process = active_count < self.max_concurrent_per_model
                
                if can_process:
                    # Get next request for this model
                    next_request = session.query(RequestQueue).filter(
                        and_(
                            RequestQueue.model_id == model_id,
                            RequestQueue.status == QueueStatus.QUEUED.value
                        )
                    ).order_by(
                        desc(RequestQueue.priority),
                        RequestQueue.created_at
                    ).first()
                    
                    if next_request:
                        # Mark as processing
                        next_request.start_processing()
                        with self._lock:
                            self._active_requests[model_name] = active_count + 1
                        session.commit()
                        
                        # Process in background thread to avoid blocking
                        processing_thread = threading.Thread(
                            target=self._process_request,
                            args=(next_request.id,),
                            name=f"process_request_{next_request.id}",
                            daemon=True
                        )
                        processing_thread.start()
    
    def _process_request(self, queue_item_id: int):
        """Process a single request."""
        try:
            db_manager = get_database_manager()
            with db_manager.get_session() as session:
                queue_item = session.query(RequestQueue).filter(RequestQueue.id == queue_item_id).first()
                if not queue_item:
                    logger.error(f"Queue item {queue_item_id} not found")
                    return
                
                request_data = queue_item.get_request_data()
                request_id = request_data["request_id"]
                model_name = queue_item.model.name
                
                logger.info(f"Processing inference request {request_id} for model {model_name}")
                
                # Create inference request record
                inference_request = InferenceRequest(
                    request_id=request_id,
                    session_id=queue_item.session_id,
                    model_id=queue_item.model_id,
                    status=InferenceStatus.PROCESSING.value
                )
                inference_request.set_input_data(request_data)
                session.add(inference_request)
                session.commit()
                
                # Get callback
                callback = self._pending_callbacks.pop(request_id, None)
                
                try:
                    # Process the request based on type
                    if request_data["request_type"] == "text":
                        result = self._process_text_request(model_name, request_data)
                    elif request_data["request_type"] == "transcription":
                        result = self._process_transcription_request(model_name, request_data)
                    elif request_data["request_type"] == "tts":
                        result = self._process_tts_request(request_data)
                    elif request_data["request_type"] == "embeddings":
                        result = self._process_embeddings_request(model_name, request_data)
                    elif request_data["request_type"] == "vision_generation":
                        result = self._process_vision_request(model_name, request_data)
                    else:
                        raise ValueError(f"Unknown request type: {request_data['request_type']}")
                    
                    # Mark as completed
                    inference_request.mark_completed(result)
                    queue_item.status = QueueStatus.COMPLETED.value
                    session.commit()
                    
                    # Call callback with success
                    if callback:
                        try:
                            if request_data.get("streaming"):
                                # For streaming, callback should be called with generator
                                callback(request_id, True, result)
                            else:
                                callback(request_id, True, result)
                        except Exception as e:
                            logger.error(f"Error in completion callback: {e}")
                    
                    logger.info(f"Completed inference request {request_id}")
                    
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Error processing request {request_id}: {error_msg}")
                    
                    # Mark as failed
                    inference_request.mark_failed(error_msg)
                    queue_item.status = QueueStatus.FAILED.value
                    session.commit()
                    
                    # Call callback with error
                    if callback:
                        try:
                            callback(request_id, False, error_msg)
                        except Exception as callback_error:
                            logger.error(f"Error in error callback: {callback_error}")
                
                finally:
                    # Decrement active count
                    with self._lock:
                        current_active = self._active_requests.get(model_name, 0)
                        if current_active > 0:
                            self._active_requests[model_name] = current_active - 1
                
        except Exception as e:
            logger.error(f"Error in request processing: {e}")
    
    def _process_text_request(self, model_name: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a text generation request."""
        from .model_manager import get_model_manager
        
        model_manager = get_model_manager()
        config = GenerationConfig(**request_data["config"])
        
        # Use asyncio.run to handle async generation in thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                model_manager.generate_text(model_name, request_data["prompt"], config)
            )
            return {
                "text": result.text,
                "total_tokens": result.total_tokens,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "generation_time_seconds": result.generation_time_seconds,
                "tokens_per_second": result.tokens_per_second
            }
        finally:
            loop.close()
    
    def _process_transcription_request(self, model_name: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process an audio transcription request."""
        from .model_manager import get_model_manager
        
        model_manager = get_model_manager()
        loaded_model = model_manager.get_model_for_inference(model_name)
        if not loaded_model:
            raise RuntimeError(f"Model {model_name} is not loaded")
        
        audio_data = request_data.get("audio_data", {})
        file_path = audio_data.get("file_path")
        if not file_path:
            raise ValueError("No audio file path provided")
        
        result = loaded_model.mlx_wrapper.transcribe_audio(
            file_path,
            language=audio_data.get("language"),
            initial_prompt=audio_data.get("initial_prompt"),
            temperature=audio_data.get("temperature", 0.0)
        )
        
        if isinstance(result, dict) and 'text' in result:
            return {"text": result['text']}
        elif isinstance(result, str):
            return {"text": result}
        else:
            return {"text": str(result)}
    
    def _process_tts_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a text-to-speech request."""
        # Implementation depends on TTS library being used
        # For now, return a placeholder
        return {"audio_file": "generated_audio.wav", "duration_seconds": 5.0}
    
    def _process_embeddings_request(self, model_name: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process an embeddings request."""
        from .model_manager import get_model_manager
        
        model_manager = get_model_manager()
        loaded_model = model_manager.get_model_for_inference(model_name)
        if not loaded_model:
            raise RuntimeError(f"Model {model_name} is not loaded")
        
        # Implementation depends on embedding generation method
        # For now, return a placeholder
        return {"embeddings": [[0.1] * 768], "dimensions": 768}
    
    def _process_vision_request(self, model_name: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a vision generation request."""
        from .model_manager import get_model_manager
        
        model_manager = get_model_manager()
        loaded_model = model_manager.get_model_for_inference(model_name)
        if not loaded_model:
            raise RuntimeError(f"Model {model_name} is not loaded")
        
        # Implementation depends on vision model capabilities
        # For now, return a placeholder
        return {"generated_image": "generated_image.png"}


# Global inference manager instance
_inference_manager: Optional[InferenceRequestManager] = None
_inference_manager_lock = threading.Lock()


def get_inference_manager() -> InferenceRequestManager:
    """Get the global inference manager instance."""
    global _inference_manager
    
    if _inference_manager is None:
        with _inference_manager_lock:
            if _inference_manager is None:
                # Read max concurrent requests per model from database
                try:
                    db_manager = get_database_manager()
                    max_concurrent = db_manager.get_setting("max_concurrent_requests_per_model", 1)
                except Exception as e:
                    logger.warning(f"Failed to read max concurrent requests setting: {e}")
                    max_concurrent = 1
                
                _inference_manager = InferenceRequestManager(max_concurrent_per_model=max_concurrent)
    
    return _inference_manager


def shutdown_inference_manager():
    """Shutdown the global inference manager."""
    global _inference_manager
    if _inference_manager:
        _inference_manager.stop()
        _inference_manager = None