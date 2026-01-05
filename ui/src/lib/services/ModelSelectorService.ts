import type { SelectorService } from '$lib/types/selector-service';
import { apiClient } from './ApiClient';

interface ModelsResponse {
    models: string[];
}

class ModelSelectorService implements SelectorService {
    async getOptions(pipelineType: string): Promise<string[]> {
        try {
            const endpoint = `/models?pipeline_type=${encodeURIComponent(pipelineType)}`;
            const response = await apiClient.get<ModelsResponse>(endpoint);
            return response.models;
        } catch (error) {
            console.error('Failed to fetch models:', error);
            // Fallback to empty array
            return [];
        }
    }
}

export const modelSelectorService = new ModelSelectorService();
