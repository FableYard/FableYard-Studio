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
        "adapters": None  # optional
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
        max_new_tokens = payload.get("maxNewTokens", 512)  # Default to 512 for txt2txt
        image_width = payload.get("imageWidth", 512)
        image_height = payload.get("imageHeight", 512)
        seed = payload.get("seed", -1)  # Default to random
        adapters = payload.get("adapters")
        adapter_dir = payload.get("adapter_dir")

        # Handle seed: clamp and generate random if -1
        if seed == -1:
            import random
            seed = random.randint(0, 2147483647)
            print(f"[WORKER] Generated random seed: {seed}")
        else:
            # Clamp to valid range
            seed = max(0, min(2147483647, seed))
            print(f"[WORKER] Using specified seed: {seed}")

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

        # Resolve adapter paths if adapters are provided
        if adapters and adapter_dir:
            from pathlib import Path
            resolved_adapters = {}
            print(f"[WORKER DEBUG] Resolving adapter paths with base dir: {adapter_dir}")
            for adapter_key, adapter_info in adapters.items():
                relative_path = adapter_info["path"]  # e.g., "flux/lora1.safetensors"
                full_path = Path(adapter_dir) / relative_path
                resolved_adapters[adapter_key] = {
                    "path": str(full_path),
                    "strength": adapter_info["strength"]
                }
                print(f"[WORKER DEBUG] Resolved adapter '{adapter_key}': {relative_path} -> {full_path}")
            adapters = resolved_adapters
        elif adapters:
            print("[WORKER WARNING] Adapters provided but adapter_dir missing - skipping path resolution")

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
        guidance_scale = 3.5
        batch_size = 1

        try:
            # Create pipeline using factory
            print(f"[WORKER DEBUG] Creating pipeline with prompts={repr(prompts)}")

            # Build type-specific params
            if pipeline_type == "txt2img":
                print(f"[WORKER DEBUG] Using {step_count} diffusion steps")
                print(f"[WORKER DEBUG] Image size: {image_width}x{image_height}")
                params = {
                    "batch_size": batch_size,
                    "adapters": adapters,
                    "step_count": step_count,
                    "image_height": image_height,
                    "image_width": image_width,
                    "guidance_scale": guidance_scale,
                    "image_name": job_id,
                }
            elif pipeline_type == "txt2txt":
                print(f"[WORKER DEBUG] Using max_new_tokens={max_new_tokens}")
                params = {
                    "max_new_tokens": max_new_tokens,
                }
            else:
                params = {}

            pipeline = Pipeline.create(
                pipeline_type=pipeline_type,
                model_family=model_family,
                model_name=model_name,
                prompts=prompts,
                seed=seed,
                params=params,
            )

            # Execute pipeline (synchronous, blocking)
            # No asyncio.to_thread needed - worker runs in separate process
            result = pipeline.execute()

            # Handle output based on pipeline type
            if pipeline_type == "txt2txt":
                # txt2txt returns generated text directly
                return {
                    "generated_text": result,
                    "job_id": job_id,
                    "pipeline_type": pipeline_type,
                    "model_family": model_family,
                    "model_name": model_name,
                    "status": "completed"
                }
            else:
                # txt2img returns file path
                from pathlib import Path
                filename = Path(result).name
                image_url = f"/api/outputs/{filename}"
                return {
                    "result_path": result,
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
