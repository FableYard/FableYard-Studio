export type Result<T, E = string> =
    | { ok: true; value: T }
    | { ok: false; error: E };

export interface SelectorService {
    getOptions(payload: string): Promise<Result<string[]>>
}
