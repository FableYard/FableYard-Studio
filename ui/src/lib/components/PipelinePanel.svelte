<script lang="ts">
import Button from "$lib/components/Button.svelte";

interface PipelinePanelProps {
    onPipelineSelect: (pipelineType: string) => void
}

let { onPipelineSelect }: PipelinePanelProps = $props();

const pipelineTypes = [
    "Image to Text",
    "Image to Image",
    "Image to Video",
    "Image to Audio",
    "Text to Text",
    "Text to Image",
    "Text to Video",
    "Text to Audio",
    "Video to Text",
    "Video to Image",
    "Video to Video",
    "Video to Audio",
    "Audio to Text",
    "Audio to Image",
    "Audio to Video",
    "Audio to Audio"
];
</script>

<nav id="pipeline-panel" class="panel">
    {#each pipelineTypes as type}
        <Button label={type} onClick={() => onPipelineSelect(type)} class="pipeline-button"></Button>
    {/each}
</nav>

<style>
    #pipeline-panel {
        display: flex;
        flex-direction: column;
        gap: 0;
        padding: 0;
        background-color: #34495e;
        min-width: 200px;
        max-width: 200px;
        overflow-y: auto;
    }

    #pipeline-panel :global(.pipeline-button) {
        height: 3.75rem;
        font-weight: bold;
        border: 1px solid white;
    }
</style>
