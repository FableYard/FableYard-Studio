from fastapi import APIRouter, Query
from typing import Optional

from models import AdaptersResponse, ModelsResponse, MediaResponse, MediaRequest, OutputsResponse
from services.storage import StorageService
from services.queue import QueueService

def create_router(
        storage_service: StorageService,
        queue_service: QueueService
) -> APIRouter:
    router = APIRouter()

    @router.get("/adapters", response_model=AdaptersResponse)
    async def list_adapters(model_family: Optional[str] = Query(None)):
        """Get a list of locally stored adapter filenames."""
        adapters = storage_service.get_adapters(model_family=model_family)
        return AdaptersResponse(adapters=adapters)

    @router.get("/models", response_model=ModelsResponse)
    async def list_models(pipeline_type: Optional[str] = Query(None)):
        """Get a list of locally stored models filenames, optionally filtered by pipeline type."""
        models = storage_service.get_models(pipeline_type=pipeline_type)
        return ModelsResponse(models=models)

    @router.get("/outputs", response_model=OutputsResponse)
    async def list_outputs():
        """Get a list of output images ordered by the most recent first."""
        outputs = storage_service.get_outputs()
        return OutputsResponse(outputs=outputs)

    @router.post("/media", response_model=MediaResponse)
    async def create_media(request: MediaRequest):
        """Queue a request and return job information."""
        response = queue_service.queue_job(request)  # No await - synchronous now
        return response

    return router
