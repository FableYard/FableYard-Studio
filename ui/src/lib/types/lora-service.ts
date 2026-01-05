export type Result<T, E = string> =
    | { ok: true; value: T }
    | { ok: false; error: E };

export interface LoRAService {
    getLoRAs(): Promise<Result<string[]>>
}