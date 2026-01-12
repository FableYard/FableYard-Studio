import type { Result } from '$lib/types/selector-service';
import { apiClient } from './ApiClient';
import type {SelectorService} from "$lib/types/selector-service";

interface AdaptersResponse {
    adapters: string[];
}

class AdapterSelectorService implements SelectorService {
    async getOptions(modelFamily: string): Promise<Result<string[]>> {
        try {
            const response = await apiClient.get<AdaptersResponse>(
                `/adapters?model_family=${encodeURIComponent(modelFamily)}`
            );
            return { ok: true, value: response.adapters };
        } catch (err) {
            return { ok: false, error: `Error loading Adapters: ${String(err)}` };
        }
    }
}

export const adapterSelectorService = new AdapterSelectorService();
