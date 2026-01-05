const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws';

type EventCallback = (event: any) => void;

class WebSocketService {
    private ws: WebSocket | null = null;
    private callbacks: EventCallback[] = [];
    private reconnectInterval: number = 5000;
    private reconnectTimer: number | null = null;

    connect() {
        if (this.ws?.readyState === WebSocket.OPEN) {
            return;
        }

        this.ws = new WebSocket(WS_URL);

        this.ws.onopen = () => {
            console.log('WebSocket connected');
            if (this.reconnectTimer) {
                clearTimeout(this.reconnectTimer);
                this.reconnectTimer = null;
            }
        };

        this.ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.callbacks.forEach(callback => callback(data));
            } catch (error) {
                console.error('Error parsing WebSocket message:', error);
            }
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };

        this.ws.onclose = () => {
            console.log('WebSocket disconnected, reconnecting...');
            this.reconnectTimer = window.setTimeout(() => {
                this.connect();
            }, this.reconnectInterval);
        };
    }

    disconnect() {
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }

    subscribe(callback: EventCallback) {
        this.callbacks.push(callback);
        return () => {
            this.callbacks = this.callbacks.filter(cb => cb !== callback);
        };
    }
}

export const webSocketService = new WebSocketService();
