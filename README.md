# MLX-GUI

A lightweight RESTful wrapper around Apple's MLX engine for dynamically loading and serving MLX-compatible models.

## Features

- REST API server with `/v1` endpoints for model management and inference
- Dynamic model loading/unloading with memory checking
- Web GUI for browsing and managing models
- SQLite database for state persistence
- System tray integration for macOS
- Multi-user inference queue system
- HuggingFace model integration
- Multimodal support (text, audio, image)

## Requirements

- Python 3.11+
- Apple Silicon (M1/M2/M3) for optimal performance
- MLX-LM v0.25.1+

## Installation

```bash
pip install mlx-gui
```

## Usage

```bash
mlx-gui start --port 8000
```

## Development

```bash
git clone https://github.com/ramborogers/mlx-gui.git
cd mlx-gui
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## License

MIT