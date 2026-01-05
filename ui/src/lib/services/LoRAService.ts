import type { LoRAService, Result } from '$lib/types/lora-service';
import { apiClient } from './ApiClient';

interface LorasResponse {
    loras: string[];
}

class LoRAServiceImpl implements LoRAService {
    async getLoRAs(): Promise<Result<string[]>> {
        try {
            const response = await apiClient.get<LorasResponse>('/loras');
            return { ok: true, value: response.loras };
        } catch (err) {
            return { ok: false, error: `Error loading LoRAs: ${String(err)}` };
        }
    }
}

export const loraService = new LoRAServiceImpl();
