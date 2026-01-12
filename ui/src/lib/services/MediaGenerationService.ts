import { apiClient } from './ApiClient';

export interface MediaRequest {
    pipelineType: string;
    model: string;
    prompts: Record<string, Record<string, string>>;  // Flexible prompt structure: {"encoder": {"positive": "...", "negative": "..."}}
    stepCount?: number;
    imageWidth?: number;
    imageHeight?: number;
    seed?: number;
    adapters?: Record<string, Record<string, string | number>>; // {"adapter_key": {"path": string, "strength": number}}
}

export interface MediaResponse {
    runId: string;
    timestamp: string;
    queuePosition: number;
    pipelineType: string;
    model: string;
}

class MediaGenerationService {
    async generateMedia(request: MediaRequest): Promise<MediaResponse> {
        return await apiClient.post<MediaRequest, MediaResponse>('/media', request);
    }
}

export const mediaGenerationService = new MediaGenerationService();
