<script lang="ts">
    import { statusService } from '$lib/services/StatusService';
    import type { CompleteStatus } from '$lib/types/status-service';

    let generatedText = $state<string | null>(null);

    // Reactively subscribe to status updates
    $effect(() => {
        const unsubscribe = statusService.statuses.subscribe(statuses => {
            // Find the most recent complete status with generated text
            for (const status of statuses) {
                if (status.type === 'complete') {
                    const completeStatus = status as CompleteStatus;
                    if (completeStatus.generatedText) {
                        generatedText = completeStatus.generatedText;
                        break;
                    }
                }
            }
        });
        return unsubscribe;
    });
</script>

<div id="output-text-panel" class="panel">
    <div class="panel-header">Output</div>
    <div class="output-container">
        {#if generatedText}
            <div class="generated-text">{generatedText}</div>
        {:else}
            <div class="placeholder">Generated text will appear here...</div>
        {/if}
    </div>
</div>

<style>
    #output-text-panel {
        display: flex;
        flex-direction: column;
        padding: 1rem;
        height: 100%;
        box-sizing: border-box;
    }

    .panel-header {
        font-weight: bold;
        margin-bottom: 0.5rem;
        font-size: 1rem;
        color: #2c3e50;
    }

    .output-container {
        flex: 1;
        border: 1px solid #ccc;
        border-radius: 4px;
        padding: 1rem;
        background-color: #f9f9f9;
        overflow-y: auto;
        min-height: 200px;
    }

    .generated-text {
        white-space: pre-wrap;
        word-wrap: break-word;
        font-family: inherit;
        line-height: 1.6;
        color: #333;
    }

    .placeholder {
        color: #999;
        font-style: italic;
    }
</style>
