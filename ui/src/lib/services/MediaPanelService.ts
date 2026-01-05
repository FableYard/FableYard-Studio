import { apiClient } from './ApiClient';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface OutputImage {
    filename: string;
    timestamp: number;
}

interface OutputsResponse {
    outputs: OutputImage[];
}

class MediaPanelService {
    /**
     * Get output images ordered by most recent first.
     * Contains only business logic - no UI updates.
     */
    async getOutputs(): Promise<OutputImage[]> {
        try {
            const response = await apiClient.get<OutputsResponse>('/outputs');
            // Images are already sorted by the API (newest first)
            return response.outputs;
        } catch (error) {
            console.error('Failed to fetch outputs:', error);
            return [];
        }
    }

    /**
     * Get the URL for an output image by filename.
     */
    getImageUrl(filename: string): string {
        return `${API_BASE_URL}/api/outputs/${filename}`;
    }
}

export const mediaPanelService = new MediaPanelService();
