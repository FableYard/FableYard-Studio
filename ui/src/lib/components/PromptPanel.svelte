<script lang="ts">
    import type {ModelPromptService, PromptConfig} from "$lib/types/model-prompt-service";
    import { onMount } from 'svelte';

    interface PromptPanelProps {
        service: ModelPromptService;
        pipelineType: string;
        modelFamily: string;
        modelName: string;
    }

    let { service, pipelineType, modelFamily, modelName }: PromptPanelProps = $props();
    let promptConfig: PromptConfig | null = $state(null);
    let error: string | null = $state(null);
    let promptValues: Record<string, string> = $state({});

    async function loadPrompts() {
        try {
            error = null;
            promptConfig = await service.getModelPrompts(pipelineType, modelFamily, modelName);
            // Reset prompt values when config changes
            promptValues = {};
        } catch (e) {
            error = e instanceof Error ? e.message : 'Failed to load prompts';
            promptConfig = null;
        }
    }

    // Expose method to get current prompt values
    export function getPromptValues(): Record<string, string> {
        return { ...promptValues };
    }

    onMount(() => {
        loadPrompts();
    });

    $effect(() => {
        if (pipelineType && modelFamily && modelName) {
            loadPrompts();
        }
    });
</script>

<div id="prompt-panel" class="panel">
    {#if error}
        <div class="error">{error}</div>
    {:else if promptConfig}
        {#each Object.entries(promptConfig) as [promptName, promptTypes]}
            <div class="prompt-card">
                <div class="card-header">
                    <h3 class="card-title">{promptName}</h3>
                </div>
                <div class="card-body">
                    {#each promptTypes as promptType}
                        <div class="prompt-field">
                            <label for="{promptName}-{promptType}">{promptType}</label>
                            <textarea
                                id="{promptName}-{promptType}"
                                placeholder="Enter {promptType} prompt..."
                                bind:value={promptValues[`${promptName}-${promptType}`]}
                            ></textarea>
                        </div>
                    {/each}
                </div>
            </div>
        {/each}
    {/if}
</div>

<style>
    .panel {
        display: flex;
        flex-direction: column;
        gap: 1rem;
        padding: 1rem;
    }

    .error {
        color: red;
        padding: 1rem;
        border: 1px solid red;
        border-radius: 4px;
        background-color: #fee;
    }

    .prompt-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        background-color: white;
    }

    .card-header {
        background-color: #f5f5f5;
        padding: 0.75rem 1rem;
        border-bottom: 1px solid #ddd;
    }

    .card-title {
        margin: 0;
        font-size: 1.125rem;
        font-weight: 600;
    }

    .card-body {
        padding: 1rem;
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }

    .prompt-field {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
    }

    .prompt-field label {
        font-weight: 500;
        font-size: 0.875rem;
        text-transform: capitalize;
    }

    .prompt-field textarea {
        min-height: 100px;
        padding: 0.5rem;
        border: 1px solid #ccc;
        border-radius: 4px;
        font-family: inherit;
        font-size: 0.875rem;
        resize: vertical;
    }

    .prompt-field textarea:focus {
        outline: none;
        border-color: #4a90e2;
        box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.1);
    }
</style>