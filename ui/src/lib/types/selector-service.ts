export interface SelectorService {
    getOptions(pipelineType: string): Promise<string[]>
}
