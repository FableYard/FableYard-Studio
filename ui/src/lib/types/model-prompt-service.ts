export type PromptType = 'positive' | 'negative'

export type PromptConfig = {
    [promptName: string]: PromptType[]
}

export type ModelConfig = {
    [modelName: string]: PromptConfig
}

export type ModelFamilyConfig = {
    [modelFamily: string]: ModelConfig
}

export type ModelPromptConfig = {
    [pipelineType: string]: ModelFamilyConfig
}

export interface ModelPromptService {
    getModelPrompts(
        pipelineType: string,
        modelFamily: string,
        modelName: string
    ): Promise<PromptConfig>
}
