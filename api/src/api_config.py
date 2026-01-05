from pathlib import Path

# Default user paths for models, loras, and outputs
# Paths are relative to the project root (FableYard-Studio/)
LORAS_DIR = Path(__file__).parent.parent.parent / "user" / "loras"
MODELS_DIR = Path(__file__).parent.parent.parent / "user" / "models"
OUTPUTS_DIR = Path(__file__).parent.parent.parent / "user" / "output"
