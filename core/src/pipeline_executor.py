# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Pipeline Executor

Handles async execution of ML pipelines with job tracking.
Factory pattern for dynamic pipeline instantiation based on model type/family/name.
"""

import sys
import os
from pathlib import Path
from threading import Thread, Lock
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import traceback

import torch

# Setup paths
script_dir = Path(__file__).parent
core_dir = script_dir.parent
os.chdir(core_dir)
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

# Model path resolution
from utils.model_resolver import resolve_model_path

# Lazy torch import - only imported when Pipeline.create() is called
def _enable_tf32():
    """Enable TF32 for faster inference on Ampere+ GPUs (RTX 30 series and newer)"""
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

@dataclass
class PromptData:
    """Single prompt with type and positive/negative text"""
    prompt_type: str  # "clip", "t5", "clip local", "clip global", etc.
    positive: str
    negative: str


@dataclass
class JobStatus:
    """Job execution status"""
    job_id: str
    status: str  # "queued", "running", "completed", "failed"
    result_path: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization"""
        return {
            "job_id": self.job_id,
            "status": self.status,
            "result_path": self.result_path,
            "error": self.error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class Pipeline:
    """
    Factory class for creating pipeline instances.
    Routes to the correct monolithic pipeline based on pipeline_type/model_family/model_name.
    """

    @staticmethod
    def create(
        pipeline_type: str,
        model_family: str,
        model_name: str,
        batch_size: int,
        # clip_prompt: str,
        # t5_prompt: str,
        prompts: dict[str, dict[str, str]],
        adapters: dict[str, dict[str, str | float]] | None,
        step_count: int,
        image_height: int,
        image_width: int,
        seed: int,
        guidance_scale: float,
        image_name: str | None = None
    ):
        """
        Factory method to create the correct pipeline instance.

        Args:
            pipeline_type: Type of pipeline (e.g., "txt2img")
            model_family: Model family (e.g., "flux", "z", "stablediffusion")
            model_name: Model version (e.g., "dev.0.30.0", "turbo")
            batch_size: Number of images to generate in parallel
            prompts: Dict mapping prompt types to positive/negative text
                     Example: {"clip": {"positive": "...", "negative": "..."},
                               "t5": {"positive": "...", "negative": "..."}}
            adapters: Dict mapping adapter paths and their strengths
            step_count: Number of diffusion steps
            image_height: Output image height in pixels
            image_width: Output image width in pixels
            seed: Random seed for reproducibility
            guidance_scale: Guidance scale for classifier-free guidance
            image_name: Optional filename for saved image (without extension)

        Returns:
            Pipeline instance (FluxPipeline, ZImagePipeline, etc.)

        Raises:
            ValueError: If pipeline_type or model_family is unsupported
            FileNotFoundError: If model path doesn't exist
            NotImplementedError: If pipeline is not yet implemented
        """
        # Enable TF32 for better performance (imports torch internally)
        _enable_tf32()

        # Resolve model path using the model_resolver
        model_path = resolve_model_path(pipeline_type, model_family, model_name)

        # Route to correct pipeline implementation
        if pipeline_type == "txt2img":
            model_family_lower = model_family.lower()

            if model_family_lower == "flux":
                from pipelines.txt2img.flux.flux import FluxPipeline
                return FluxPipeline(
                    model_path=model_path,
                    batch_size=batch_size,
                    # clip_prompt=clip_prompt,
                    # t5_prompt=t5_prompt,
                    adapters=adapters,
                    prompts=prompts,
                    step_count=step_count,
                    image_height=image_height,
                    image_width=image_width,
                    seed=seed,
                    guidance_scale=guidance_scale,
                    image_name=image_name
                )

            elif model_family_lower in ["stablediffusion", "stable_diffusion"]:
                # Placeholder for future StableDiffusionPipeline implementation
                raise NotImplementedError(
                    f"StableDiffusionPipeline not yet implemented. "
                    f"Create pipelines/txt2img/stable_diffusion/stable_diffusion.py"
                )

            elif model_family_lower == "z": # TODO: Change model family to "z_image"
                from pipelines.txt2img.z_image.z_image import ZImagePipeline
                return ZImagePipeline(
                    model_path=model_path,
                    batch_size=batch_size,
                    # clip_prompt=clip_prompt,
                    # t5_prompt=t5_prompt,
                    prompts=prompts,
                    adapters=adapters,
                    step_count=step_count,
                    image_height=image_height,
                    image_width=image_width,
                    seed=seed,
                    guidance_scale=guidance_scale,
                    image_name=image_name
                )

            elif model_family_lower == "pony":
                # Placeholder for future PonyPipeline implementation
                raise NotImplementedError(
                    f"PonyPipeline not yet implemented. "
                    f"Create pipelines/txt2img/pony/pony.py"
                )

            else:
                raise ValueError(
                    f"Unsupported model_family '{model_family}' for pipeline_type '{pipeline_type}'. "
                    f"Supported families: flux, z, stablediffusion, pony"
                )

        else:
            raise ValueError(
                f"Unsupported pipeline_type: '{pipeline_type}'. "
                f"Supported types: txt2img"
            )


class PipelineExecutor:
    """
    Singleton executor for ML pipelines with async job management.
    Uses the Pipeline factory to instantiate the correct pipeline dynamically.
    """

    _instance = None
    _lock = Lock()

    def __init__(self):
        if PipelineExecutor._instance is not None:
            raise RuntimeError("Use get_instance() to access PipelineExecutor")

        self.jobs: Dict[str, JobStatus] = {}
        self.job_lock = Lock()

    @classmethod
    def get_instance(cls) -> 'PipelineExecutor':
        """Get singleton instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def create_job(
        self,
        pipeline_type: str,
        model_family: str,
        model_name: str,
        step_count: int,
        prompts: List[Dict[str, str]],
        job_id: str,
        image_height: int = 512,
        image_width: int = 512,
        seed: int = 42,
        guidance_scale: float = 3.5,
        batch_size: int = 1
    ) -> Dict[str, Any]:
        """
        Create and queue a new pipeline job.

        Args:
            pipeline_type: Type of pipeline (e.g., "txt2img")
            model_family: Model family (e.g., "flux", "stablediffusion")
            model_name: Model version (e.g., "dev.0.30.0")
            step_count: Number of diffusion steps
            prompts: List of prompt dicts with {prompt_type, positive, negative}
            job_id: Unique job identifier
            image_height: Output image height (default: 512)
            image_width: Output image width (default: 512)
            seed: Random seed (default: 42)
            guidance_scale: Guidance scale (default: 3.5)
            batch_size: Batch size (default: 1)

        Returns:
            Dict with job_id and status
        """
        # Validate inputs
        if not prompts or len(prompts) == 0:
            raise ValueError("At least one prompt is required")

        if step_count < 1 or step_count > 1000:
            raise ValueError("step_count must be between 1 and 1000")

        # Convert prompts to PromptData objects
        prompt_objects = [
            PromptData(
                prompt_type=p["prompt_type"],
                positive=p["positive"],
                negative=p["negative"]
            )
            for p in prompts
        ]

        # Create job status
        job_status = JobStatus(
            job_id=job_id,
            status="queued",
            created_at=datetime.now()
        )

        with self.job_lock:
            self.jobs[job_id] = job_status

        info(f"Created job {job_id}: {pipeline_type}/{model_family}/{model_name}, {step_count} steps")

        # Start background execution thread
        thread = Thread(
            target=self._execute_pipeline_thread,
            args=(
                job_id,
                pipeline_type,
                model_family,
                model_name,
                step_count,
                prompt_objects,
                image_height,
                image_width,
                seed,
                guidance_scale,
                batch_size
            ),
            daemon=True
        )
        thread.start()

        return {
            "job_id": job_id,
            "status": "queued"
        }

    def _execute_pipeline_thread(
        self,
        job_id: str,
        pipeline_type: str,
        model_family: str,
        model_name: str,
        step_count: int,
        prompts: List[PromptData],
        image_height: int,
        image_width: int,
        seed: int,
        guidance_scale: float,
        batch_size: int
    ):
        """
        Background thread for pipeline execution.
        Uses Pipeline factory to instantiate and execute the correct pipeline.
        """
        try:
            # Update status to running
            with self.job_lock:
                self.jobs[job_id].status = "running"

            # Build prompts dict with nested structure
            prompts_dict = {
                p.prompt_type.lower(): {
                    "positive": p.positive,
                    "negative": p.negative
                }
                for p in prompts
            }

            # Create pipeline using factory
            pipeline = Pipeline.create(
                pipeline_type=pipeline_type,
                model_family=model_family,
                model_name=model_name,
                batch_size=batch_size,
                prompts=prompts_dict,
                step_count=step_count,
                image_height=image_height,
                image_width=image_width,
                seed=seed,
                guidance_scale=guidance_scale,
                image_name=job_id
            )

            # Execute pipeline
            result_path = pipeline.execute()

            # Update status to completed
            with self.job_lock:
                self.jobs[job_id].status = "completed"
                self.jobs[job_id].result_path = result_path
                self.jobs[job_id].completed_at = datetime.now()

            info(f"Job {job_id} completed successfully: {result_path}")

        except Exception as e:
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            error(f"Job {job_id} failed: {error_msg}")

            with self.job_lock:
                self.jobs[job_id].status = "failed"
                self.jobs[job_id].error = str(e)
                self.jobs[job_id].completed_at = datetime.now()

    def execute_pipeline_with_events(
        self,
        pipeline_type: str,
        model_family: str,
        model_name: str,
        step_count: int,
        prompts: List[Dict[str, str]],
        job_id: str,
        image_height: int = 512,
        image_width: int = 512,
        seed: int = 42,
        guidance_scale: float = 3.5,
        batch_size: int = 1
    ):
        """
        Execute pipeline synchronously with SSE event streaming.
        Yields pipeline-level events: job_started, job_complete, job_error.

        Args:
            pipeline_type: Type of pipeline (e.g., "txt2img")
            model_family: Model family (e.g., "flux", "stablediffusion")
            model_name: Model version (e.g., "dev.0.30.0")
            step_count: Number of diffusion steps
            prompts: List of prompt dicts with {prompt_type, positive, negative}
            job_id: Unique job identifier
            image_height: Output image height (default: 512)
            image_width: Output image width (default: 512)
            seed: Random seed (default: 42)
            guidance_scale: Guidance scale (default: 3.5)
            batch_size: Batch size (default: 1)

        Yields:
            SSE formatted events
        """
        import json as json_module

        try:
            # Create job entry with 'running' status
            job_status = JobStatus(
                job_id=job_id,
                status="running",
                created_at=datetime.now()
            )

            with self.job_lock:
                self.jobs[job_id] = job_status

            # Yield job started event
            yield f"data: {json_module.dumps({'type': 'job_started', 'data': {'job_id': job_id}})}\n\n"

            # Validate inputs
            if not prompts or len(prompts) == 0:
                raise ValueError("At least one prompt is required")

            if step_count < 1 or step_count > 1000:
                raise ValueError("step_count must be between 1 and 1000")

            # Build prompts dict with nested structure
            prompts_dict = {
                p["prompt_type"].lower(): {
                    "positive": p["positive"],
                    "negative": p["negative"]
                }
                for p in prompts
            }

            # Create pipeline using factory
            pipeline = Pipeline.create(
                pipeline_type=pipeline_type,
                model_family=model_family,
                model_name=model_name,
                batch_size=batch_size,
                prompts=prompts_dict,
                step_count=step_count,
                image_height=image_height,
                image_width=image_width,
                seed=seed,
                guidance_scale=guidance_scale,
                image_name=job_id
            )

            # Execute pipeline
            result_path = pipeline.execute()

            # Update job status to completed
            with self.job_lock:
                self.jobs[job_id].status = "completed"
                self.jobs[job_id].result_path = result_path
                self.jobs[job_id].completed_at = datetime.now()

            # Send completion event with image URL
            image_url = f'/v1/pipelines/jobs/{job_id}/image'
            yield f"data: {json_module.dumps({'type': 'job_complete', 'data': {'job_id': job_id, 'image_url': image_url, 'result_path': result_path}})}\n\n"

        except Exception as e:
            import traceback
            error_data = {
                "job_id": job_id,
                "error": str(e),
                "traceback": traceback.format_exc()
            }

            # Update job status to failed
            with self.job_lock:
                if job_id in self.jobs:
                    self.jobs[job_id].status = "failed"
                    self.jobs[job_id].error = str(e)
                    self.jobs[job_id].completed_at = datetime.now()

            yield f"data: {json_module.dumps({'type': 'job_error', 'data': error_data})}\n\n"

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific job"""
        with self.job_lock:
            job = self.jobs.get(job_id)
            return job.to_dict() if job else None

    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all jobs"""
        with self.job_lock:
            return [job.to_dict() for job in self.jobs.values()]
