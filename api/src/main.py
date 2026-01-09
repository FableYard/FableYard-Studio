"""
FableYard Studio API

FastAPI backend for communication between UI and Worker.
Uses multiprocessing.Queue for job queueing and events.
"""

import sys
import asyncio
import multiprocessing as mp
from pathlib import Path
from contextlib import asynccontextmanager

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Add project root for shared imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from routers.media import create_router
from routers.websocket import create_websocket_router
from services.storage import StorageService
from services.queue import QueueService
from services.websocket_manager import WebSocketManager
from api_config import ADAPTER_DIR, MODELS_DIR, OUTPUTS_DIR

from shared.queue_bridge import JobQueue, EventBridge


def create_app(job_queue: mp.Queue, event_queue: mp.Queue) -> FastAPI:
    """Create FastAPI app with multiprocessing queues"""

    # Initialize services
    storage_service = StorageService(
        adapter_dir=ADAPTER_DIR,
        models_dir=MODELS_DIR,
        outputs_dir=OUTPUTS_DIR
    )

    # Wrap multiprocessing queues
    job_service = JobQueue(job_queue)
    event_bridge = EventBridge(event_queue)

    # Create queue service
    queue_service = QueueService(job_service, event_bridge, ADAPTER_DIR)

    # Create WebSocket manager
    ws_manager = WebSocketManager()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan context manager - startup and shutdown"""
        # Startup
        print("[API] Initializing services...")

        # Start WebSocket event listener (polls event_queue)
        event_task = asyncio.create_task(
            ws_manager.listen_to_events(event_bridge)
        )

        print("[API] WebSocket manager started")
        print("[API] Ready to accept requests\n")

        yield

        # Shutdown
        print("\n[API] Shutting down services...")
        await ws_manager.stop()
        event_task.cancel()
        try:
            await event_task
        except asyncio.CancelledError:
            pass
        print("[API] Shutdown complete")

    # Create app with lifespan
    app = FastAPI(
        title="FableYard Studio API",
        description="API for communication between UI and Worker",
        version="2.0.0",
        lifespan=lifespan
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers with injected services
    media_router = create_router(storage_service, queue_service)
    app.include_router(media_router)

    websocket_router = create_websocket_router(ws_manager)
    app.include_router(websocket_router)

    # Mount static files for serving output images
    app.mount("/api/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

    @app.get("/")
    async def root():
        return {
            "message": "FableYard Studio API is running",
            "version": "0.1.1",
            "queue_type": "multiprocessing"
        }

    return app


# For direct uvicorn execution (legacy support)
if __name__ == "__main__":
    print("[ERROR] Please use start.py to launch the application")
    print("Usage: python start.py")
    sys.exit(1)
