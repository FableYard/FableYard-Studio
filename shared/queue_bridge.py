"""
Queue Bridge - Multiprocessing Queue Wrappers

Provides compatible interface for job queueing and event publishing.
"""

import multiprocessing as mp
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import queue


class JobQueue:
    """Job queue wrapper for multiprocessing.Queue - replaces AsyncQueue"""

    def __init__(self, queue: mp.Queue):
        self._queue = queue

    def enqueue(self, request_id: str, task_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Add a job to the queue"""
        task = {
            "id": request_id,
            "type": task_type,
            "payload": payload,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        self._queue.put(task)
        return task

    def dequeue(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Get next job from queue (blocking)"""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError("No job available in queue")

    def is_empty(self) -> bool:
        """Check if queue is empty"""
        return self._queue.empty()

    def qsize(self) -> int:
        """Get approximate queue size"""
        try:
            return self._queue.qsize()
        except NotImplementedError:
            # qsize() not implemented on macOS
            return 0


class EventBridge:
    """Event bridge for publishing events - replaces EventBus"""

    def __init__(self, queue: mp.Queue):
        self._queue = queue

    def publish(self, event_type: str, data: Dict[str, Any]) -> None:
        """Publish an event to the event queue"""
        event = {
            "type": event_type,
            "data": data
        }
        self._queue.put(event)

    def get_event(self, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        """Get next event from queue (non-blocking)"""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
