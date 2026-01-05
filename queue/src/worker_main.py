"""
Worker Process Main Entry Point

Polls job_queue for tasks and processes them using PipelineWorker.
Publishes events to event_queue for real-time WebSocket updates.
"""

import sys
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Any

# Add queue/src to path for absolute imports
QUEUE_SRC = Path(__file__).parent
CORE_SRC = QUEUE_SRC.parent.parent / "core" / "src"
PROJECT_ROOT = QUEUE_SRC.parent.parent

sys.path.insert(0, str(QUEUE_SRC))
sys.path.insert(0, str(CORE_SRC))
sys.path.insert(0, str(PROJECT_ROOT))

from shared.queue_bridge import JobQueue, EventBridge
from workers.pipeline_worker import PipelineWorker


def worker_main(job_queue: mp.Queue, event_queue: mp.Queue):
    """
    Worker process main loop

    Args:
        job_queue: Multiprocessing queue for receiving jobs
        event_queue: Multiprocessing queue for publishing events
    """
    # Wrap queues
    job_service = JobQueue(job_queue)
    event_bridge = EventBridge(event_queue)

    # Create worker instance
    worker = PipelineWorker(event_bridge)

    print("\n" + "=" * 50)
    print("  FableYard Worker Process")
    print("  Ready to process pipeline jobs")
    print("=" * 50 + "\n")
    print("[Worker] Waiting for jobs from queue...\n")

    while True:
        try:
            # Block until job available (1 second timeout for graceful shutdown)
            try:
                task = job_service.dequeue(timeout=1.0)
            except TimeoutError:
                # No job available, continue loop
                continue

            task_id = task.get("id", "unknown")
            task_type = task.get("type", "unknown")

            print(f"\n[Worker] Received job: {task_id}")
            print(f"[Worker] Task type: {task_type}")

            # Publish processing event (include both task_id and job_id for compatibility)
            event_bridge.publish("task.processing", {
                "task_id": task_id,
                "job_id": task_id,
                "task_type": task_type,
                "status": "processing"
            })

            # Process task (blocking ML inference - takes ~11 minutes)
            try:
                import time
                start_time = time.time()

                result = worker.process_sync(task)

                # Calculate duration in milliseconds
                duration_ms = int((time.time() - start_time) * 1000)

                # Publish completion event with proper structure for StatusService
                event_bridge.publish("task.completed", {
                    "task_id": task_id,
                    "job_id": task_id,
                    "task_type": task_type,
                    "status": "completed",
                    "result": {
                        "output_path": result.get("result_path", ""),
                        "image_url": result.get("image_url", ""),
                        "duration": duration_ms
                    }
                })

                print(f"[Worker] ✓ Job {task_id} completed successfully in {duration_ms}ms\n")

            except Exception as e:
                # Publish failure event
                error_msg = str(e)
                event_bridge.publish("task.failed", {
                    "task_id": task_id,
                    "job_id": task_id,
                    "task_type": task_type,
                    "status": "failed",
                    "error": error_msg
                })

                print(f"[Worker] ✗ Job {task_id} failed: {error_msg}\n")
                import traceback
                traceback.print_exc()

        except KeyboardInterrupt:
            print("\n[Worker] Received shutdown signal")
            break
        except Exception as e:
            print(f"[Worker] Unexpected error: {e}")
            import traceback
            traceback.print_exc()

    print("[Worker] Shutting down...")
