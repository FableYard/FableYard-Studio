"""Pipeline worker implementation - Synchronous version for multiprocessing"""

from typing import Dict, Any


class PipelineWorker:
    """
    Worker that executes ML pipelines using the core Pipeline factory.
    Synchronous version for use with multiprocessing (no async/await).

    Payload structure:
    {
        "pipelineType": "txt2img",
        "model": "flux/dev.0.30.0",  # format: <family>/<name>
        "prompts": {  # flexible prompt structure based on model
            "clip": {"positive": "A beautiful sunset"},
            "t5": {"positive": "A beautiful sunset over mountains"}
        },
        "stepCount": 20,  # number of diffusion steps (default: 20)
        "imageWidth": 512,  # image width in pixels (default: 512)
        "imageHeight": 512,  # image height in pixels (default: 512)
        "lora": None  # optional
    }
    """

    def __init__(self, event_bridge):
        """
        Initialize pipeline worker

        Args:
            event_bridge: EventBridge instance for publishing progress events
        """
        self.event_bridge = event_bridge

    def process_sync(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a pipeline synchronously (blocking)

        Args:
            task: Task dictionary containing payload and metadata

        Returns:
            Dict with result_path and job metadata

        Raises:
            RuntimeError: If pipeline execution fails
            ValueError: If payload is invalid
        """
        payload = task.get("payload", {})

        # Extract and validate task parameters
        pipeline_type = payload.get("pipelineType")
        model = payload.get("model")
        prompts = payload.get("prompts")
        step_count = payload.get("stepCount", 20)  # Default to 20 if not provided
        image_width = payload.get("imageWidth", 512)
        image_height = payload.get("imageHeight", 512)
        lora = payload.get("lora")

        # Debug: Print what we received
        print(f"[WORKER DEBUG] Received prompts: {repr(prompts)}")
        print(f"[WORKER DEBUG] Step count: {step_count}")
        print(f"[WORKER DEBUG] Image dimensions: {image_width}x{image_height}")

        if not pipeline_type or not model or not prompts:
            raise ValueError(
                f"Missing required parameters. Got: pipelineType={pipeline_type}, "
                f"model={model}, prompts={prompts}"
            )

        # Validate step count
        if step_count < 1:
            raise ValueError(
                f"Invalid step count: {step_count}. Step count must be at least 1."
            )

        # Parse model string: "flux/dev.0.30.0" -> family="flux", name="dev.0.30.0"
        try:
            model_parts = model.split("/")
            if len(model_parts) != 2:
                raise ValueError(
                    f"Invalid model format: '{model}'. Expected format: <family>/<name> "
                    f"(e.g., 'flux/dev.0.30.0')"
                )
            model_family, model_name = model_parts
        except Exception as e:
            raise ValueError(f"Failed to parse model '{model}': {e}")

        # Import Pipeline factory from core
        try:
            from pipeline_executor import Pipeline
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import Pipeline from core layer. "
                f"Ensure core/src is accessible. Error: {e}"
            )

        # Generate unique job ID for output
        job_id = payload.get("job_id", "worker_output")

        # Pipeline parameters with defaults
        # TODO: Make these configurable via payload or config
        seed = 42
        guidance_scale = 3.5
        batch_size = 1

        try:
            # Create pipeline using factory
            print(f"[WORKER DEBUG] Creating pipeline with prompts={repr(prompts)}")
            print(f"[WORKER DEBUG] Using {step_count} diffusion steps")
            print(f"[WORKER DEBUG] Image size: {image_width}x{image_height}")
            pipeline = Pipeline.create(
                pipeline_type=pipeline_type,
                model_family=model_family,
                model_name=model_name,
                batch_size=batch_size,
                prompts=prompts,
                step_count=step_count,
                image_height=image_height,
                image_width=image_width,
                seed=seed,
                guidance_scale=guidance_scale,
                image_name=job_id
            )

            # Execute pipeline (synchronous, blocking ~11 minutes)
            # No asyncio.to_thread needed - worker runs in separate process
            result_path = pipeline.execute()

            # Extract filename for image URL
            from pathlib import Path
            filename = Path(result_path).name
            image_url = f"/api/outputs/{filename}"

            return {
                "result_path": result_path,
                "image_url": image_url,
                "job_id": job_id,
                "pipeline_type": pipeline_type,
                "model_family": model_family,
                "model_name": model_name,
                "status": "completed"
            }

        except NotImplementedError as e:
            raise RuntimeError(
                f"Pipeline not implemented: {pipeline_type}/{model_family}. "
                f"Error: {e}"
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Model not found: {pipeline_type}/{model_family}/{model_name}. "
                f"Ensure model is installed in user/models/. Error: {e}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Pipeline execution failed for {pipeline_type}/{model_family}/{model_name}. "
                f"Error: {e}"
            )
