<script lang="ts">
    import type { SelectorService } from '$lib/types/selector-service';

    interface Placeholder {
        label: string
        id: string
    }

    interface SelectorProps {
        service: SelectorService
        pipelineType?: string
        modelFamily?: string
        placeholder: Placeholder
        onModelSelect?: (modelFamily: string, modelName: string) => void
        onAdapterSelect?: (adapterName: string) => void
    }
    let {
        service,
        pipelineType,
        modelFamily,
        placeholder,
        onModelSelect,
        onAdapterSelect
    }: SelectorProps = $props();
    let options = $state<string[]>([]);

    $effect(() => {
        // Reset options when dependencies change
        options = [];

        if (pipelineType) {
            service.getOptions(pipelineType).then(result => {
                if (result.ok) {
                    options = result.value;
                } else {
                    options = [];
                    console.error(`Failed to load model options: ${result.error}`);
                }
            })
        } else if (modelFamily) {
            service.getOptions(modelFamily).then(result => {
                if (result.ok) {
                    options = result.value;
                } else {
                    options = [];
                    console.error(`Failed to load adapter options: ${result.error}`);
                }
            })
        }
    });

    function handleModelSelect(event: Event) {
        const target = event.currentTarget as HTMLSelectElement;
        const [modelFamily, modelName] = target.value.split('/');

        if (onModelSelect) {
            onModelSelect(modelFamily, modelName);
        }
    }

    function handleAdapterSelect(event: Event) {
        const target = event.currentTarget as HTMLSelectElement;
        const adapterName = target.value;

        if (onAdapterSelect) {
            onAdapterSelect(adapterName);
        }
    }

    function handleChange(event: Event) {
        if (onAdapterSelect) {
            handleAdapterSelect(event);
        } else if (onModelSelect) {
            handleModelSelect(event);
        }
    }
</script>

<label for="{placeholder.id}-selector">{placeholder.label}</label>
<select id="{placeholder.id}-selector" class="selector" onchange={handleChange}>
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