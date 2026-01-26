import type { ModelPromptService, ModelPromptConfig, PromptConfig } from '$lib/types/model-prompt-service';

class ModelPromptServiceImpl implements ModelPromptService {
    private config: ModelPromptConfig | null = null;
    private configPromise: Promise<ModelPromptConfig> | null = null;

    // Map UI pipeline names to config keys
    private pipelineTypeMap: Record<string, string> = {
        "Image to Audio": "img2aud",
        "Image to Image": "img2img",
        "Image to Text": "img2txt",
        "Image to Video": "img2vid",
        "Text to Audio": "txt2aud",
        "Text to Image": "txt2img",
        "Text to Text": "txt2txt",
        "Text to Video": "txt2vid",
    };

    private async loadConfig(): Promise<ModelPromptConfig> {
        if (this.config) {
            return this.config;
        }

        if (!this.configPromise) {
            this.configPromise = fetch('/model-prompt-config.json')
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`Failed to load config: ${response.statusText}`);
                    }
                    return response.json();
                })
                .then(data => {
                    this.config = data;
                    return data;
                });
        }

        return this.configPromise;
    }

    async getModelPrompts(pipelineType: string, modelFamily: string, modelName: string): Promise<PromptConfig> {
        const config = await this.loadConfig();

        // Convert UI pipeline type to config key
        const configPipelineType = this.pipelineTypeMap[pipelineType] || pipelineType;

        const pipelineConfig = config[configPipelineType];
        if (!pipelineConfig) {
            // If txt2txt, return default config (all txt2txt models use same prompt structure)
            if (configPipelineType === 'txt2txt') {
                return { "Text": ["positive"] };
            }
            throw new Error(`Pipeline type "${pipelineType}" (${configPipelineType}) not found in configuration`);
        }

        const familyConfig = pipelineConfig[modelFamily];
        if (!familyConfig) {
            // If txt2txt and model not found, return default config
            if (configPipelineType === 'txt2txt') {
                return { "Text": ["positive"] };
            }
            throw new Error(`Model family "${modelFamily}" not found for pipeline "${pipelineType}"`);
        }

        const modelConfig = familyConfig[modelName];
        if (!modelConfig) {
            // If txt2txt and model not found, return default config
            if (configPipelineType === 'txt2txt') {
                return { "Text": ["positive"] };
            }
            throw new Error(`Model "${modelName}" not found in family "${modelFamily}"`);
        }

        return modelConfig;
    }
}

export const modelPromptService = new ModelPromptServiceImpl();
