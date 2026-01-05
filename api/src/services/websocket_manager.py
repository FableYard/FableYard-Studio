"""
WebSocket Manager

Manages WebSocket connections and broadcasts events from worker to UI.
Uses multiprocessing.Queue.
"""

import asyncio
from typing import Set
from fastapi import WebSocket
import json


class WebSocketManager:
    """Manages WebSocket connections and event broadcasting"""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._running = False

    async def connect(self, websocket: WebSocket):
        """Accept and register a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        print(f"[WebSocket] Client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Unregister a WebSocket connection"""
        self.active_connections.discard(websocket)
        print(f"[WebSocket] Client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """
        Broadcast message to all connected WebSocket clients

        Args:
            message: Dictionary to broadcast (will be JSON-encoded)
        """
        if not self.active_connections:
            return

        message_json = json.dumps(message)
        disconnected = set()

        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
            except Exception as e:
                print(f"[WebSocket] Error sending to client: {e}")
                disconnected.add(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

    async def listen_to_events(self, event_bridge):
        """
        Poll event_queue and broadcast events to WebSocket clients

        Args:
            event_bridge: EventBridge instance wrapping multiprocessing.Queue

        This replaces Redis pub/sub with polling from multiprocessing.Queue
        """
        self._running = True
        print("[WebSocket] Event listener started")

        try:
            while self._running:
                # Poll for events (non-blocking with short timeout)
                event = event_bridge.get_event(timeout=0.1)

                if event:
                    # EventBridge wraps events as {"type": "...", "data": {...}}
                    # Extract the data portion and broadcast it to WebSocket clients
                    event_data = event.get("data", event)
                    await self.broadcast(event_data)

                # Small sleep to prevent busy loop
                await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            print("[WebSocket] Event listener cancelled")
        except Exception as e:
            print(f"[WebSocket] Event listener error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._running = False

    async def stop(self):
        """Stop the event listener and close all connections"""
        print("[WebSocket] Stopping...")
        self._running = False

        # Close all active connections
        for connection in list(self.active_connections):
            try:
                await connection.close()
            except:
                pass

        self.active_connections.clear()
        print("[WebSocket] Stopped")
