# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Component-based Pipeline Runner
Executes pipeline using component abstraction with automatic SSE events.
"""
from typing import Dict, Any, List, Generator
from components.base_component import PipelineComponent
from utils.event_emitter import EventEmitter


class PipelineRunner:
    """
    Executes a pipeline by running components in sequence.
    Each component automatically gets lifecycle management and SSE events.
    """

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.components: List[PipelineComponent] = []

    def add_component(self, component: PipelineComponent):
        """Add a component to the pipeline"""
        self.components.append(component)

    def execute(self, initial_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute all components in sequence WITHOUT SSE streaming.
        Each component's output becomes the next component's input.
        """
        context = initial_inputs.copy()

        for component in self.components:
            with component:
                outputs = component.execute(context)
                context.update(outputs)

        return context

    def execute_with_events(self, initial_inputs: Dict[str, Any]) -> Generator[str, None, Dict[str, Any]]:
        """
        Execute all components in sequence WITH SSE streaming.
        Yields SSE events as components execute.

        Usage:
            runner = PipelineRunner("job-123")
            runner.add_component(TokenizerComponent(...))
            runner.add_component(TransformerComponent(...))

            # Returns generator that yields SSE events
            for event in runner.execute_with_events({"prompt": "..."}):
                yield event  # Forward to Flask Response
        """
        # Initialize EventEmitter for this job
        emitter = EventEmitter.get_instance()
        emitter.initialize(self.job_id, total_steps=len(self.components))

        try:
            # Start event generator in background
            import threading
            context = initial_inputs.copy()
            error = None

            def run_pipeline():
                nonlocal context, error
                try:
                    for component in self.components:
                        with component:
                            outputs = component.execute(context)
                            context.update(outputs)
                except Exception as e:
                    error = e
                finally:
                    emitter.finalize()

            # Start pipeline execution in thread
            thread = threading.Thread(target=run_pipeline, daemon=False)
            thread.start()

            # Yield SSE events as they're emitted
            for event in emitter.get_event_generator():
                yield event

            # Wait for pipeline to complete
            thread.join()

            # Raise error if one occurred
            if error:
                raise error

            return context

        except Exception as e:
            emitter.emit_error(str(e))
            emitter.finalize()
            raise


# Example usage in refactored pipeline_executor.py:
"""
def execute_pipeline(self, job_id: str, pipeline_name: str, step_count: int, prompts):
    # Create pipeline runner
    runner = PipelineRunner(job_id)

    model_path = Path("models/flux/dev.0.30.0")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build pipeline from config
    runner.add_component(TokenizerComponent({}, model_path, device))
    runner.add_component(EncoderComponent({}, model_path, device))
    runner.add_component(SchedulerComponent({"step_count": step_count}, model_path, device))
    runner.add_component(TransformerComponent({}, model_path, device))
    runner.add_component(DiffusionComponent({"step_count": step_count}, model_path, device))
    runner.add_component(VAEComponent({}, model_path, device))
    runner.add_component(ImageSaverComponent({"job_id": job_id}, model_path, device))

    # Execute pipeline - each component emits SSE events automatically
    result = runner.execute({
        "prompts": {
            "clip": {"positive": prompts[0].positive, "negative": prompts[0].negative},
            "t5": {"positive": prompts[1].positive, "negative": prompts[1].negative}
        }
    })

    return result["output_path"]
"""
