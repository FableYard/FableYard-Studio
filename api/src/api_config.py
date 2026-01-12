from pathlib import Path

# Default user paths for models, adapters, and outputs
# Paths are relative to the project root (FableYard-Studio/)
ADAPTER_DIR = Path(__file__).parent.parent.parent / "user" / "adapters"
MODELS_DIR = Path(__file__).parent.parent.parent / "user" / "models"
OUTPUTS_DIR = Path(__file__).parent.parent.parent / "user" / "output"
