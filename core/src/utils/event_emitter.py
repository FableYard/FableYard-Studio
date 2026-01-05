# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

"""
EventEmitter Singleton for Server-Sent Events (SSE)
Thread-safe singleton that manages event streaming for pipeline execution.
"""
import json
from threading import Lock
from typing import Optional, Dict, Any
from queue import Queue


class EventEmitter:
    """
    Singleton class for emitting SSE events throughout the pipeline.
    Components use this to send real-time progress updates.
    """
    _instance = None
    _lock = Lock()

    def __init__(self):
        if EventEmitter._instance is not None:
            raise RuntimeError("Use EventEmitter.get_instance() instead")
        self.event_queue: Optional[Queue] = None
        self.current_job_id: Optional[str] = None
        self.total_steps: int = 0
        self.current_step: int = 0

    @classmethod
    def get_instance(cls) -> 'EventEmitter':
        """Get singleton instance (thread-safe)"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def initialize(self, job_id: str, total_steps: int):
        """Initialize event emitter for a new job"""
        self.current_job_id = job_id
        self.total_steps = total_steps
        self.current_step = 0
        self.event_queue = Queue()

    def emit_component_start(self, component: str):
        """Emit component start event"""
        self.current_step += 1
        event = {
            "type": "component_start",
            "data": {
                "component": component,
                "step": self.current_step,
                "total": self.total_steps,
                "job_id": self.current_job_id
            }
        }
        self._emit(event)

    def emit_component_complete(self, component: str):
        """Emit component complete event"""
        event = {
            "type": "component_complete",
            "data": {
                "component": component,
                "step": self.current_step,
                "total": self.total_steps,
                "job_id": self.current_job_id
            }
        }
        self._emit(event)

    def emit_progress(self, message: str, progress: float):
        """Emit progress event (for long-running operations like diffusion loop)"""
        event = {
            "type": "progress",
            "data": {
                "message": message,
                "progress": progress,
                "step": self.current_step,
                "job_id": self.current_job_id
            }
        }
        self._emit(event)

    def emit_diffusion_progress(self, current_iteration: int, total_iterations: int, component: str = "Transformer"):
        """Emit diffusion iteration progress event"""
        progress_percent = (current_iteration / total_iterations) * 100
        event = {
            "type": "diffusion_progress",
            "data": {
                "component": component,
                "current_iteration": current_iteration,
                "total_iterations": total_iterations,
                "progress": progress_percent,
                "step": self.current_step,
                "job_id": self.current_job_id
            }
        }
        self._emit(event)

    def emit_error(self, error: str, traceback: Optional[str] = None):
        """Emit error event"""
        event = {
            "type": "job_error",
            "data": {
                "job_id": self.current_job_id,
                "error": error,
                "traceback": traceback
            }
        }
        self._emit(event)

    def _emit(self, event: Dict[str, Any]):
        """Internal: put event in queue"""
        if self.event_queue is not None:
            self.event_queue.put(event)

    def get_event_generator(self):
        """Generator that yields SSE-formatted events"""
        if self.event_queue is None:
            return

        while True:
            event = self.event_queue.get()
            if event is None:  # Sentinel to stop
                break
            yield f"data: {json.dumps(event)}\n\n"

    def finalize(self):
        """Signal end of event stream"""
        if self.event_queue is not None:
            self.event_queue.put(None)  # Sentinel
        self.current_job_id = None
        self.event_queue = None
        self.current_step = 0

    @property
    def is_active(self) -> bool:
        """Check if emitter is currently active"""
        return self.event_queue is not None
