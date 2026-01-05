# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Base Component Interface for Pipeline Execution
All pipeline components (tokenizers, encoders, transformers, etc.) inherit from this.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pathlib import Path
import torch
import gc

# Enable TF32 for faster inference on Ampere+ GPUs (RTX 30 series and newer)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class PipelineComponent(ABC):
    """
    Abstract base class for all pipeline components.
    Provides lifecycle hooks, automatic resource management, and SSE event emission.

    Components automatically emit:
    - component_start event when load() is called
    - component_complete event when execute() finishes
    """

    def __init__(self, component_path: Path, device: str):
        self.component_path = component_path
        self.device = device
        self._model = None

    @property
    @abstractmethod
    def component_name(self) -> str:
        """Return the display name for SSE events (e.g., 'CLIPTokenizer')"""
        pass

    @abstractmethod
    def load(self) -> None:
        """
        Load model/resources into memory.
        Called once before execute().

        NOTE: You don't need to emit SSE events manually -
        the base class handles this automatically.
        """
        pass

    @abstractmethod
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute component logic with given inputs.
        Returns outputs as dict to pass to next component.

        NOTE: You don't need to emit SSE events manually -
        the base class handles this automatically.
        """
        pass

    def cleanup(self) -> None:
        """
        Release resources and clear GPU memory.
        Called after execute() completes.
        """
        if self._model is not None:
            # Move to CPU before deletion to properly free GPU memory
            try:
                if hasattr(self._model, 'to'):
                    self._model = self._model.to('cpu')
            except Exception:
                pass  # Model may not support .to() method

            del self._model
            self._model = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _emit_start_event(self):
        """Internal: Emit component_start SSE event"""
        try:
            from utils.event_emitter import EventEmitter
            emitter = EventEmitter.get_instance()
            if emitter.is_active:
                emitter.emit_component_start(self.component_name)
        except Exception:
            # If EventEmitter not initialized, silently continue
            pass

    def _emit_complete_event(self):
        """Internal: Emit component_complete SSE event"""
        try:
            from utils.event_emitter import EventEmitter
            emitter = EventEmitter.get_instance()
            if emitter.is_active:
                emitter.emit_component_complete(self.component_name)
        except Exception:
            # If EventEmitter not initialized, silently continue
            pass

    def __enter__(self):
        """Context manager support: auto-load and emit start event"""
        # Clear memory before loading this component
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._emit_start_event()
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager support: emit complete event and auto-cleanup"""
        if exc_type is None:
            # Only emit complete if no exception occurred
            self._emit_complete_event()
        self.cleanup()
        return False
