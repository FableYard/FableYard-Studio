from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from services.websocket_manager import WebSocketManager


def create_websocket_router(ws_manager: WebSocketManager) -> APIRouter:
    router = APIRouter()

    @router.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await ws_manager.connect(websocket)
        try:
            # Keep connection alive, wait for messages
            while True:
                # Receive messages from client (ping/pong, etc.)
                data = await websocket.receive_text()
                # Echo back or handle client messages if needed
                # For now, this is primarily server->client broadcast
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket)

    return router
