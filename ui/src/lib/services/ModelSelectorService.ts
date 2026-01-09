import type { Result, SelectorService } from '$lib/types/selector-service';
import { apiClient } from './ApiClient';
import {writable} from "svelte/store";

interface ModelsResponse {
    models: string[];
}

class ModelSelectorService implements SelectorService {
    private _selectedModelFamily = writable<string | null>(null);
    selectedModelFamily = this._selectedModelFamily;

    async getOptions(pipelineType: string): Promise<Result<string[]>> {
        try {
            const response = await apiClient.get<ModelsResponse>(
                `/models?pipeline_type=${encodeURIComponent(pipelineType)}`
            );
            return { ok: true, value: response.models };
        } catch (err) {
            return { ok: false, error: `Error loading Models: ${String(err)}` };
        }
    }

    async modelChangeDetected(modelFamily: string) {
        this._selectedModelFamily.set(modelFamily);
    }
}

export const modelSelectorService = new ModelSelectorService();
