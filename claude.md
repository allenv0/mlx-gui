# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MLX-GUI is a lightweight RESTful wrapper around Apple's MLX engine for dynamically loading and serving MLX-compatible models. The project includes a REST API server, web GUI, system tray integration, and comprehensive model management capabilities.

## Current Status

This project is in early development phase - the core architecture is documented but implementation is minimal. The project needs foundational setup including proper Python packaging, dependency management, and core module structure.

## Key Architecture Components

- **REST API Server**: `/v1` endpoints for model management and inference
- **Model Management**: Dynamic loading/unloading with memory checking
- **Web GUI**: Interface for browsing and managing models
- **SQLite Database**: State persistence in user directory
- **System Tray**: macOS integration for easy access
- **Queue System**: Multi-user inference handling

## Development Setup

Since this is early-stage development, initial setup will likely require:

1. **Python Environment**: Requires Python 3.11+ with MLX dependencies
2. **Package Structure**: Standard Python package with `pyproject.toml` or `setup.py`
3. **Dependencies**: Core dependencies likely include MLX, FastAPI/Flask, SQLite, GUI framework
4. **CLI Interface**: Main entry point should be `mlx-gui start --port 8000`

## API Design

The project follows a RESTful pattern with these key endpoints:
- `POST /v1/models/{model_name}/load` - Load models with path and version
- `POST /v1/models/{model_name}/generate` - Generate inference with multimodal inputs
- `GET /v1/models/{model_name}/health` - Health checks

## Memory Management

A critical feature is checking system RAM before loading models with clear error messages like "This model requires 480GB of RAM you have 16GB". This should be implemented early in the model loading pipeline.

## Important Considerations

- **Apple Silicon Focus**: Optimized for Apple Silicon with MLX-LM>=0.24.0
- **Multimodal Support**: Handle text, audio, and image inputs via MLX vision add-ons  
- **HuggingFace Integration**: Enumerate and pull models using HuggingFace API with MLX tags
- **User Experience**: Emphasis on beautiful GUI and intuitive tray icon interface
- **State Management**: SQLite database should be stored in standard user application directory

## Development Workflow

When implementing features:
1. Start with core API endpoints and model loading logic
2. Add memory checking and error handling early
3. Implement database schema for model state persistence
4. Build GUI components after API is stable
5. Add system tray integration for macOS
6. Include comprehensive error messages and user feedback