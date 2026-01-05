# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Service Layer HTTP Server

Lightweight Flask server that wraps PipelineExecutor.
Exposes ML pipeline functionality via HTTP API on port 8001.
"""

from flask import Flask, request, jsonify, send_file, Response, stream_with_context
import sys
from pathlib import Path
import json as json_module
import torch

# Enable TF32 for faster inference on Ampere+ GPUs (RTX 30 series and newer)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Add src to path for imports
script_dir = Path(__file__).parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from pipeline_executor import PipelineExecutor

app = Flask(__name__)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "ML Pipeline Service"})


@app.route('/v1/pipelines/<pipeline_type>/<model_family>/<model_name>', methods=['POST'])
def create_pipeline_job(pipeline_type, model_family, model_name):
    """
    Create and execute a pipeline job with SSE streaming.

    Streams Server-Sent Events for pipeline-level progress:
    - event: job_started
    - event: job_complete (with image URL)
    - event: job_error

    Expected request body:
    {
        "job_id": "unique_job_id",
        "step_count": 15,
        "prompts": [
            {"prompt_type": "clip", "positive": "...", "negative": "..."},
            {"prompt_type": "t5", "positive": "...", "negative": "..."}
        ],
        "image_height": 512,  // optional, default 512
        "image_width": 512,   // optional, default 512
        "seed": 42,           // optional, default 42
        "guidance_scale": 3.5, // optional, default 3.5
        "batch_size": 1       // optional, default 1
    }
    """
    try:
        # Parse request body
        data = request.get_json()

        if not data:
            return jsonify({"error": "Request body is required"}), 400

        # Validate required fields
        required_fields = ['job_id', 'step_count', 'prompts']
        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            return jsonify({
                "error": f"Missing required fields: {', '.join(missing_fields)}"
            }), 400

        # Validate prompts is a list
        if not isinstance(data['prompts'], list):
            return jsonify({"error": "prompts must be an array"}), 400

        if len(data['prompts']) == 0:
            return jsonify({"error": "At least one prompt is required"}), 400

        # Get PipelineExecutor instance
        executor = PipelineExecutor.get_instance()

        # Extract optional parameters with defaults
        image_height = data.get('image_height', 512)
        image_width = data.get('image_width', 512)
        seed = data.get('seed', 42)
        guidance_scale = data.get('guidance_scale', 3.5)
        batch_size = data.get('batch_size', 1)

        # Create SSE stream generator
        def generate():
            try:
                # Execute pipeline with event streaming
                for event in executor.execute_pipeline_with_events(
                    pipeline_type=pipeline_type,
                    model_family=model_family,
                    model_name=model_name,
                    step_count=data['step_count'],
                    prompts=data['prompts'],
                    job_id=data['job_id'],
                    image_height=image_height,
                    image_width=image_width,
                    seed=seed,
                    guidance_scale=guidance_scale,
                    batch_size=batch_size
                ):
                    yield event

            except Exception as e:
                import traceback
                error_msg = str(e)
                error_json = json_module.dumps({
                    "type": "job_error",
                    "data": {"error": error_msg, "traceback": traceback.format_exc()}
                })
                yield f"data: {error_json}\n\n"

        # Return SSE stream
        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )

    except ValueError as e:
        # Validation errors
        return jsonify({"error": str(e)}), 400

    except Exception as e:
        # Unexpected errors
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"Error creating job: {error_msg}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


@app.route('/v1/pipelines/jobs/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """
    Get status of a specific job.

    Returns:
    {
        "job_id": "...",
        "status": "queued|running|completed|failed",
        "result_path": "/path/to/output.png" (if completed),
        "error": "error message" (if failed)
    }
    """
    try:
        executor = PipelineExecutor.get_instance()
        job_status = executor.get_job_status(job_id)

        if job_status is None:
            return jsonify({"error": f"Job {job_id} not found"}), 404

        return jsonify(job_status), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/v1/pipelines/jobs', methods=['GET'])
def list_jobs():
    """List all jobs"""
    try:
        executor = PipelineExecutor.get_instance()
        jobs = executor.list_jobs()
        return jsonify({"jobs": jobs}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/v1/pipelines/jobs/<job_id>/image', methods=['GET'])
def get_job_image(job_id):
    """
    Get the generated image for a completed job.

    Returns:
        PNG image file if job is completed
        404 if job not found or image doesn't exist
        400 if job is not yet completed
    """
    try:
        executor = PipelineExecutor.get_instance()
        job_status = executor.get_job_status(job_id)

        if job_status is None:
            return jsonify({"error": f"Job {job_id} not found"}), 404

        if job_status['status'] != 'completed':
            return jsonify({
                "error": f"Job is not completed (current status: {job_status['status']})"
            }), 400

        if not job_status.get('result_path'):
            return jsonify({"error": "Job completed but no result path found"}), 500

        # Get the image path
        image_path = Path(job_status['result_path'])

        if not image_path.exists():
            return jsonify({"error": f"Image file not found at {image_path}"}), 404

        # Serve the image file
        return send_file(
            image_path,
            mimetype='image/png',
            as_attachment=False,
            download_name=f'{job_id}.png'
        )

    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        print(f"Error serving image: {error_msg}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("ML Pipeline Service")
    print("=" * 60)
    print(f"Starting server on http://localhost:8001")
    print("Endpoints:")
    print("  GET  /health")
    print("  POST /v1/pipelines/<pipeline_type>/<model_family>/<model_name>")
    print("  GET  /v1/pipelines/jobs/<job_id>")
    print("  GET  /v1/pipelines/jobs/<job_id>/image")
    print("  GET  /v1/pipelines/jobs")
    print("=" * 60)
    print("Example: POST /v1/pipelines/txt2img/flux/dev.0.30.0")
    print("=" * 60)

    # Run Flask server
    app.run(
        host='localhost',
        port=8001,
        debug=False,  # Disable debug mode for production
        threaded=True  # Handle multiple requests concurrently
    )
