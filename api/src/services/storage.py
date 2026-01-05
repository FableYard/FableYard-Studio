from pathlib import Path
from typing import Optional


class StorageService:
    # Map UI pipeline names to directory names
    PIPELINE_TYPE_MAP = {
        "Image to Text": "img2txt",
        "Image to Image": "img2img",
        "Image to Video": "img2vid",
        "Image to Audio": "img2aud",
        "Text to Text": "txt2txt",
        "Text to Image": "txt2img",
        "Text to Video": "txt2vid",
        "Text to Audio": "txt2aud"
    }
    def __init__(self, loras_dir: Path, models_dir: Path, outputs_dir: Path):
        self.loras_dir = loras_dir
        self.models_dir = models_dir
        self.outputs_dir = outputs_dir

    def get_loras(self) -> list[str]:
        """Get a list of LoRA filenames from the storage directory."""
        if not self.loras_dir.exists():
            return []

        return [f.name for f in self.loras_dir.iterdir() if f.is_file()]

    def get_models(self, pipeline_type: Optional[str] = None) -> list[str]:
        """Get a list of model identifiers from the storage directory.

        Args:
            pipeline_type: Optional pipeline type to filter by (e.g., "Text to Image")
        """
        if not self.models_dir.exists():
            return []

        # Convert UI pipeline type to directory name
        pipeline_filter = None
        if pipeline_type:
            pipeline_filter = self.PIPELINE_TYPE_MAP.get(pipeline_type, pipeline_type.lower().replace(" ", ""))

        models = set()
        # Iterate through pipeline types (txt2img, img2img, etc.)
        for pipeline_dir in self.models_dir.iterdir():
            if not pipeline_dir.is_dir():
                continue

            # Skip if filtering and this isn't the target pipeline
            if pipeline_filter and pipeline_dir.name != pipeline_filter:
                continue

            # Iterate through model families (Flux, StableDiffusion, etc.)
            for family_dir in pipeline_dir.iterdir():
                if not family_dir.is_dir():
                    continue

                # Iterate through model versions (dev-1, 3-5, etc.)
                for version_dir in family_dir.iterdir():
                    if version_dir.is_dir():
                        # Use lowercase for model family to match worker expectations
                        family_name = family_dir.name.lower()
                        model_id = f"{family_name}/{version_dir.name}"
                        models.add(model_id)

        return sorted(list(models))

    def get_outputs(self) -> list[dict]:
        """Get a list of output images ordered by most recent first.

        Returns:
            List of dicts with 'filename' and 'timestamp' keys, ordered newest first
        """
        if not self.outputs_dir.exists():
            return []

        outputs = []
        # Common image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'}

        for file_path in self.outputs_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                outputs.append({
                    'filename': file_path.name,
                    'timestamp': file_path.stat().st_mtime
                })

        # Sort by timestamp descending (newest first)
        outputs.sort(key=lambda x: x['timestamp'], reverse=True)

        return outputs
