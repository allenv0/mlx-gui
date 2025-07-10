"""
Transparent queued inference wrapper that maintains OpenAI compatibility.

This module provides drop-in replacements for direct model inference calls
that automatically handle queuing when models are busy, making the queuing
completely transparent to API consumers.
"""

import asyncio
import logging
import uuid
from typing import AsyncIterator, Optional

from .inference_queue_manager import get_inference_manager, QueuedRequest
from .model_manager import get_model_manager
from .mlx_integration import GenerationConfig

logger = logging.getLogger(__name__)


async def generate_text_queued(
    model_name: str, 
    prompt: str, 
    config: GenerationConfig,
    priority: int = 5
):
    """
    Generate text with transparent queuing.
    
    This function provides the same interface as model_manager.generate_text()
    but automatically handles queuing when the model is busy.
    """
    model_manager = get_model_manager()
    inference_manager = get_inference_manager()
    
    # Check if we can process immediately
    queue_status = inference_manager.get_queue_status(model_name)
    
    if queue_status.get("can_accept_immediate", False):
        # Model is available, process directly
        logger.debug(f"Processing {model_name} request immediately")
        return await model_manager.generate_text(model_name, prompt, config)
    
    # Model is busy, use queue
    logger.debug(f"Queuing {model_name} request (active: {queue_status.get('active_requests', 0)})")
    
    session_id = str(uuid.uuid4())
    
    # Create queued request
    queued_request = QueuedRequest(
        session_id=session_id,
        model_name=model_name,
        prompt=prompt,
        config=config,
        priority=priority,
        streaming=False
    )
    
    # Use a completion future to wait for result
    result_future = asyncio.Future()
    
    def completion_callback(request_id: str, success: bool, result):
        if success:
            result_future.set_result(result)
        else:
            result_future.set_exception(Exception(result))
    
    queued_request.callback = completion_callback
    
    # Queue the request
    request_id = await inference_manager.queue_request(queued_request)
    
    # Wait for completion with timeout
    try:
        result = await asyncio.wait_for(result_future, timeout=300.0)  # 5 minute timeout
        return result
    except asyncio.TimeoutError:
        raise RuntimeError("Request timed out after 5 minutes")


async def generate_text_stream_queued(
    model_name: str, 
    prompt: str, 
    config: GenerationConfig,
    priority: int = 5
) -> AsyncIterator[str]:
    """
    Generate streaming text with transparent queuing.
    
    This function provides the same interface as model_manager.generate_text_stream()
    but automatically handles queuing when the model is busy.
    """
    model_manager = get_model_manager()
    inference_manager = get_inference_manager()
    
    # Check if we can process immediately
    queue_status = inference_manager.get_queue_status(model_name)
    
    if queue_status.get("can_accept_immediate", False):
        # Model is available, stream directly
        logger.debug(f"Streaming {model_name} request immediately")
        async for chunk in model_manager.generate_text_stream(model_name, prompt, config):
            yield chunk
        return
    
    # Model is busy, wait for turn then stream
    logger.debug(f"Queuing {model_name} streaming request (active: {queue_status.get('active_requests', 0)})")
    
    session_id = str(uuid.uuid4())
    
    # Use an event to signal when streaming can start
    stream_ready_event = asyncio.Event()
    stream_generator = None
    error_exception = None
    
    def stream_ready_callback(request_id: str, success: bool, generator_or_error):
        nonlocal stream_generator, error_exception
        if success:
            stream_generator = generator_or_error
        else:
            error_exception = Exception(generator_or_error)
        stream_ready_event.set()
    
    # Create a special queued request for streaming
    queued_request = QueuedRequest(
        session_id=session_id,
        model_name=model_name,
        prompt=prompt,
        config=config,
        priority=priority,
        streaming=True,
        stream_callback=stream_ready_callback
    )
    
    # Queue the request
    request_id = await inference_manager.queue_request(queued_request)
    
    # Wait for streaming to be ready
    try:
        await asyncio.wait_for(stream_ready_event.wait(), timeout=300.0)
    except asyncio.TimeoutError:
        raise RuntimeError("Request timed out after 5 minutes")
    
    # Check for errors
    if error_exception:
        raise error_exception
    
    # Stream the results
    if stream_generator:
        async for chunk in stream_generator:
            yield chunk


async def queued_transcribe_audio(
    model_name: str,
    file_path: str,
    language: Optional[str] = None,
    initial_prompt: Optional[str] = None,
    temperature: float = 0.0,
    priority: int = 5
):
    """
    Transcribe audio with transparent queuing.
    
    Drop-in replacement for loaded_model.mlx_wrapper.transcribe_audio() with queuing.
    """
    model_manager = get_model_manager()
    inference_manager = get_inference_manager()
    
    # Check if we can process immediately
    queue_status = inference_manager.get_queue_status(model_name)
    
    if queue_status.get("can_accept_immediate", False):
        # Model is available, process directly
        logger.debug(f"Processing {model_name} transcription immediately")
        loaded_model = model_manager.get_model_for_inference(model_name)
        if not loaded_model:
            raise RuntimeError(f"Model {model_name} is not loaded")
        
        result = loaded_model.mlx_wrapper.transcribe_audio(
            file_path,
            language=language,
            initial_prompt=initial_prompt,
            temperature=temperature
        )
        
        # Format result consistently
        if isinstance(result, dict) and 'text' in result:
            return {"text": result['text']}
        elif isinstance(result, str):
            return {"text": result}
        else:
            return {"text": str(result)}
    
    # Model is busy, use queue
    logger.debug(f"Queuing {model_name} transcription request (active: {queue_status.get('active_requests', 0)})")
    
    session_id = str(uuid.uuid4())
    
    # Create dummy config for audio requests
    dummy_config = GenerationConfig(max_tokens=1)
    
    # Create queued request
    queued_request = QueuedRequest(
        session_id=session_id,
        model_name=model_name,
        prompt="",  # Not used for transcription
        config=dummy_config,
        priority=priority,
        request_type="transcription",
        audio_data={
            "file_path": file_path,
            "language": language,
            "initial_prompt": initial_prompt,
            "temperature": temperature
        }
    )
    
    # Use a completion future to wait for result
    result_future = asyncio.Future()
    
    def completion_callback(request_id: str, success: bool, result):
        if success:
            result_future.set_result(result)
        else:
            result_future.set_exception(Exception(result))
    
    queued_request.callback = completion_callback
    
    # Queue the request
    request_id = await inference_manager.queue_request(queued_request)
    
    # Wait for completion with timeout
    try:
        result = await asyncio.wait_for(result_future, timeout=300.0)  # 5 minute timeout
        return result
    except asyncio.TimeoutError:
        raise RuntimeError("Transcription request timed out after 5 minutes")


async def queued_generate_speech(
    text: str,
    voice: str = "kokoro",
    speed: float = 1.0,
    priority: int = 5
):
    """
    Generate speech with transparent queuing.
    
    Drop-in replacement for mlx_audio.tts.generate() with queuing.
    """
    inference_manager = get_inference_manager()
    
    # For TTS, we don't need a specific model to be loaded
    # We'll use a dummy model name for queue management
    model_name = f"tts-{voice}"
    
    # Check if we can process immediately
    queue_status = inference_manager.get_queue_status(model_name)
    
    if queue_status.get("can_accept_immediate", False):
        # Process directly
        logger.debug(f"Processing TTS request immediately")
        try:
            import mlx_audio
            import tempfile
            import os
        except ImportError:
            raise RuntimeError("MLX Audio not installed")
        
        # Generate speech
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            mlx_audio.tts.generate(
                text=text,
                model=voice,
                output_file=temp_file.name,
                speed=speed
            )
            
            # Read generated audio
            with open(temp_file.name, "rb") as audio_file:
                audio_content = audio_file.read()
            
            # Clean up
            os.unlink(temp_file.name)
            
            return audio_content
    
    # Use queue for TTS
    logger.debug(f"Queuing TTS request (active: {queue_status.get('active_requests', 0)})")
    
    session_id = str(uuid.uuid4())
    
    # Create dummy config for TTS requests
    dummy_config = GenerationConfig(max_tokens=1)
    
    # Create queued request
    queued_request = QueuedRequest(
        session_id=session_id,
        model_name=model_name,
        prompt=text,
        config=dummy_config,
        priority=priority,
        request_type="tts",
        audio_data={
            "voice": voice,
            "speed": speed
        }
    )
    
    # Use a completion future to wait for result
    result_future = asyncio.Future()
    
    def completion_callback(request_id: str, success: bool, result):
        if success:
            result_future.set_result(result)
        else:
            result_future.set_exception(Exception(result))
    
    queued_request.callback = completion_callback
    
    # Queue the request
    request_id = await inference_manager.queue_request(queued_request)
    
    # Wait for completion with timeout
    try:
        result = await asyncio.wait_for(result_future, timeout=300.0)  # 5 minute timeout
        return result
    except asyncio.TimeoutError:
        raise RuntimeError("TTS request timed out after 5 minutes")


# Convenience functions that match the model_manager interface
async def queued_generate_text(model_name: str, prompt: str, config: GenerationConfig):
    """Drop-in replacement for model_manager.generate_text() with queuing."""
    return await generate_text_queued(model_name, prompt, config)


async def queued_generate_text_stream(model_name: str, prompt: str, config: GenerationConfig):
    """Drop-in replacement for model_manager.generate_text_stream() with queuing."""
    async for chunk in generate_text_stream_queued(model_name, prompt, config):
        yield chunk


async def queued_generate_embeddings(
    model_name: str,
    texts: list[str],
    priority: int = 5
):
    """
    Generate embeddings with transparent queuing.
    
    Drop-in replacement for loaded_model.mlx_wrapper.generate_embeddings() with queuing.
    """
    model_manager = get_model_manager()
    inference_manager = get_inference_manager()
    
    # Check if we can process immediately
    queue_status = inference_manager.get_queue_status(model_name)
    
    if queue_status.get("can_accept_immediate", False):
        # Model is available, process directly
        logger.debug(f"Processing {model_name} embeddings immediately")
        loaded_model = model_manager.get_model_for_inference(model_name)
        if not loaded_model:
            raise RuntimeError(f"Model {model_name} is not loaded")
        
        # Check if this is an embedding model
        if not hasattr(loaded_model.mlx_wrapper, 'generate_embeddings'):
            raise RuntimeError(f"Model {model_name} is not an embedding model")
        
        result = loaded_model.mlx_wrapper.generate_embeddings(texts)
        
        # Ensure result is in expected format
        if isinstance(result, list):
            return {
                "embeddings": result,
                "prompt_tokens": sum(len(text.split()) for text in texts),
                "total_tokens": sum(len(text.split()) for text in texts)
            }
        else:
            return result
    
    # Model is busy, use queue
    logger.debug(f"Queuing {model_name} embeddings request (active: {queue_status.get('active_requests', 0)})")
    
    session_id = str(uuid.uuid4())
    
    # Create dummy config for embedding requests
    dummy_config = GenerationConfig(max_tokens=1)
    
    # Create queued request
    queued_request = QueuedRequest(
        session_id=session_id,
        model_name=model_name,
        prompt="",  # Not used for embeddings
        config=dummy_config,
        priority=priority,
        request_type="embeddings",
        audio_data={  # Reuse audio_data field for embedding data
            "texts": texts
        }
    )
    
    # Use a completion future to wait for result
    result_future = asyncio.Future()
    
    def completion_callback(request_id: str, success: bool, result):
        if success:
            result_future.set_result(result)
        else:
            result_future.set_exception(Exception(result))
    
    queued_request.callback = completion_callback
    
    # Queue the request
    request_id = await inference_manager.queue_request(queued_request)
    
    # Wait for completion with timeout
    try:
        result = await asyncio.wait_for(result_future, timeout=300.0)  # 5 minute timeout
        return result
    except asyncio.TimeoutError:
        raise RuntimeError("Embeddings request timed out after 5 minutes")


async def queued_generate_vision(
    model_name: str, 
    prompt: str, 
    image_file_paths: list[str],
    config: GenerationConfig,
    priority: int = 5
):
    """
    Generate text with images using vision models with transparent queuing.
    
    Args:
        model_name: Name of the vision model to use
        prompt: Text prompt
        image_file_paths: List of temporary image file paths (not URLs)
        config: Generation configuration
        priority: Request priority (1-10, higher = more urgent)
        
    Returns:
        GenerationResult with text response and usage stats
    """
    inference_manager = get_inference_manager()
    
    # Check if we can process immediately
    queue_status = inference_manager.get_queue_status(model_name)
    
    if queue_status.get("can_accept_immediate", False):
        # Model is available, process directly
        model_manager = get_model_manager()
        loaded_model = model_manager.get_model_for_inference(model_name)
        
        if loaded_model:
            try:
                if hasattr(loaded_model.mlx_wrapper, 'generate_with_images'):
                    result = loaded_model.mlx_wrapper.generate_with_images(prompt, image_file_paths, config)
                    logger.debug(f"Direct vision generation for {model_name}: {len(result.text)} chars")
                    
                    # Update model usage count for direct vision generation
                    try:
                        from .database import get_database_manager
                        from .models import Model
                        db_manager = get_database_manager()
                        with db_manager.get_session() as session:
                            model_record = session.query(Model).filter(Model.name == model_name).first()
                            if model_record:
                                model_record.increment_use_count()
                                session.commit()
                                logger.debug(f"Incremented usage count for vision model {model_name}")
                    except Exception as usage_error:
                        logger.error(f"Failed to update usage count for vision model {model_name}: {usage_error}")
                    
                    return result
                else:
                    # Fallback to regular generation if not a vision model
                    result = loaded_model.mlx_wrapper.generate(prompt, config)
                    logger.debug(f"Fallback text generation for {model_name}: {len(result.text)} chars")
                    
                    # Update model usage count for direct text generation fallback
                    try:
                        from .database import get_database_manager
                        from .models import Model
                        db_manager = get_database_manager()
                        with db_manager.get_session() as session:
                            model_record = session.query(Model).filter(Model.name == model_name).first()
                            if model_record:
                                model_record.increment_use_count()
                                session.commit()
                                logger.debug(f"Incremented usage count for fallback text model {model_name}")
                    except Exception as usage_error:
                        logger.error(f"Failed to update usage count for fallback text model {model_name}: {usage_error}")
                    
                    return result
            except Exception as e:
                logger.warning(f"Direct vision generation failed for {model_name}, falling back to queue: {e}")
    
    # Model is busy or not loaded, queue the request
    logger.info(f"Queuing vision generation request for {model_name} (priority: {priority})")
    
    session_id = str(uuid.uuid4())
    result_future = asyncio.Future()
    
    # Create queued request following the established pattern
    queued_request = QueuedRequest(
        session_id=session_id,
        model_name=model_name,
        prompt=prompt,  # Use the actual prompt
        config=config,  # Use the actual config
        priority=priority,
        request_type="vision_generation",
        audio_data={  # Reuse audio_data field for vision data
            "image_file_paths": image_file_paths,  # Store file paths, not URLs
            "original_prompt": prompt  # Store original prompt in case needed
        }
    )
    
    # Set up completion callback
    def completion_callback(request_id: str, success: bool, result):
        if success:
            # Convert output_data back to GenerationResult
            if isinstance(result, dict) and result.get("type") == "vision_generation":
                from mlx_gui.mlx_integration import GenerationResult
                generation_result = GenerationResult(
                    text=result.get("text", ""),
                    prompt=result.get("prompt", ""),
                    total_tokens=result.get("total_tokens", 0),
                    prompt_tokens=result.get("prompt_tokens", 0),
                    completion_tokens=result.get("completion_tokens", 0),
                    generation_time_seconds=result.get("generation_time_seconds", 0.0),
                    tokens_per_second=result.get("tokens_per_second", 0.0)
                )
                result_future.set_result(generation_result)
            else:
                result_future.set_result(result)
        else:
            result_future.set_exception(Exception(result))
    
    queued_request.callback = completion_callback
    
    # Queue the request
    request_id = await inference_manager.queue_request(queued_request)
    
    # Wait for completion with timeout
    try:
        result = await asyncio.wait_for(result_future, timeout=300.0)  # 5 minute timeout
        return result
    except asyncio.TimeoutError:
        raise RuntimeError("Vision generation request timed out after 5 minutes")