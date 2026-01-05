"""
Queue Service

Simplified queue service using multiprocessing.Queue.
Handles job submission to the worker process.
"""

import uuid
from models import MediaRequest, MediaResponse

# Pipeline type mapping remains the same
PIPELINE_TYPE_MAP = {
    "Image to Text": "img2txt",
    "Image to Image": "img2img",
    "Image to Video": "img2vid",
    "Image to Audio": "img2aud",
    "Text to Text": "txt2txt",
    "Text to Image": "txt2img",
    "Text to Video": "txt2vid",
    "Text to Audio": "txt2aud",
    "Video to Text": "vid2txt",
    "Video to Image": "vid2img",
    "Video to Video": "vid2vid",
    "Video to Audio": "vid2aud",
    "Audio to Text": "aud2txt",
    "Audio to Image": "aud2img",
    "Audio to Video": "aud2vid",
    "Audio to Audio": "aud2aud"
}


class QueueService:
    """Queue service for submitting jobs to worker process"""

    def __init__(self, job_queue, event_bridge):
        """
        Initialize queue service with JobQueue wrapper and EventBridge

        Args:
            job_queue: JobQueue instance wrapping multiprocessing.Queue
            event_bridge: EventBridge instance for publishing events
        """
        self.job_queue = job_queue
        self.event_bridge = event_bridge

    def queue_job(self, request: MediaRequest) -> MediaResponse:
        """
        Queue a job for processing

        Args:
            request: Media generation request from UI

        Returns:
            MediaResponse with job ID and metadata
        """
        run_id = str(uuid.uuid4())

        # Map UI pipeline type to backend technical name
        technical_pipeline_type = PIPELINE_TYPE_MAP.get(
            request.pipelineType,
            request.pipelineType  # Fallback to original if not found
        )

        # Convert request to task payload
        payload = {
            "job_id": run_id,
            "pipelineType": technical_pipeline_type,
            "model": request.model,
            "prompts": request.prompts,  # Pass prompts dict as-is
            "stepCount": request.stepCount,
            "imageWidth": request.imageWidth,
            "imageHeight": request.imageHeight,
            "lora": request.lora
        }

        # Enqueue to job queue (non-blocking, instant return)
        task = self.job_queue.enqueue(
            request_id=run_id,
            task_type="media_generation",
            payload=payload
        )

        task_id = task["id"]
        timestamp = task["created_at"]

        # Get approximate queue position
        try:
            queue_position = self.job_queue.qsize()
        except:
            queue_position = 0

        # Publish queued event to WebSocket with payload structure
        self.event_bridge.publish("task.queued", {
            "task_id": task_id,
            "job_id": task_id,
            "task_type": "media_generation",
            "status": "queued",
            "payload": {
                "pipelineType": request.pipelineType,
                "model": request.model
            }
        })

        print(f"[API] Job {task_id} queued - event published")

        return MediaResponse(
            runId=task_id,
            timestamp=timestamp,
            queuePosition=queue_position,
            pipelineType=request.pipelineType,
            model=request.model
        )
