<script lang="ts">
    import type { SelectorService } from '$lib/types/selector-service';

    interface Placeholder {
        label: string
        id: string
    }

    interface SelectorProps {
        service: SelectorService
        pipelineType: string
        placeholder: Placeholder
        onModelSelect?: (modelName: string) => void
    }
    let { service, pipelineType, placeholder, onModelSelect }: SelectorProps = $props();
    let options = $state<string[]>([]);

    $effect(() => {
        const result = service.getOptions(pipelineType);
        result.then(data => {
            options = data;
        });
    });

    function handleModelSelect(event: Event) {
        const target = event.currentTarget as HTMLSelectElement;
        const modelName = target.value;
        if (modelName && onModelSelect) {
            onModelSelect(modelName);
        }
    }
</script>

<label for="{placeholder.id}-selector">{placeholder.label}</label>
<select id="{placeholder.id}-selector" class="selector" onchange={handleModelSelect}>
    <option value="" disabled selected>{placeholder.label}</option>
    {#each options as option}
        <option value={option}>{option}</option>
    {/each}
</select>

<style>
    .selector {
        width: 200px;
        min-width: 200px;
    }
</style>