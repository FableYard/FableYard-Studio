<script lang="ts">
    import Selector from './Selector.svelte';
    import { modelSelectorService } from '$lib/services/ModelSelectorService';
    import Button from "$lib/components/Button.svelte";

    interface ControlBarProps {
        pipelineType: string
        stepCount?: number
        imageDimensions?: string
        seed?: number
        onModelSelect?: (modelFamily: string, modelName: string) => void
        onStepCountChange?: (steps: number) => void
        onDimensionsChange?: (dimensions: string) => void
        onSeedChange?: (seed: number) => void
        onGenerate?: () => void
    }
    let {
        pipelineType,
        stepCount = 20,
        imageDimensions = "512x512",
        seed = -1,
        onModelSelect,
        onStepCountChange,
        onDimensionsChange,
        onSeedChange,
        onGenerate
    }: ControlBarProps = $props();

    const dimensionOptions = [
        { value: "512x512", label: "512x512" },
        { value: "512x768", label: "512x768" },
        { value: "768x512", label: "768x512" },
        { value: "768x768", label: "768x768" },
        { value: "512x1024", label: "512x1024" },
        { value: "1024x512", label: "1024x512" },
        { value: "1024x1024", label: "1024x1024" }
    ];

    function handleGenerate() {
        if (onGenerate) {
            onGenerate();
        }
    }

    function handleStepCountChange(event: Event) {
        const value = parseInt((event.target as HTMLInputElement).value);
        if (onStepCountChange && !isNaN(value)) {
            onStepCountChange(value);
        }
    }

    function handleDimensionsChange(event: Event) {
        const value = (event.target as HTMLSelectElement).value;
        if (onDimensionsChange) {
            onDimensionsChange(value);
        }
    }

    function handleSeedChange(event: Event) {
        const value = parseInt((event.target as HTMLInputElement).value);
        if (onSeedChange && !isNaN(value)) {
            onSeedChange(value);
        }
    }

    function handleModelSelect(modelFamily: string, modelName: string) {
        console.log(`Selected model: ${modelFamily}/${modelName}`);
        modelSelectorService.modelChangeDetected(modelFamily);
        onModelSelect?.(modelFamily, modelName);
    }
</script>

<div class="control-bar">
    <h3 class="control-bar__label">{pipelineType}</h3>
    <Button label="Generate" onClick={handleGenerate}/>
    <div class="control-bar__selector">
        {#if pipelineType === 'Text to Image'}
            <div class="dimension-selector">
                <label for="dimensions">Size:</label>
                <select
                    id="dimensions"
                    value={imageDimensions}
                    onchange={handleDimensionsChange}
                >
                    {#each dimensionOptions as option}
                        <option value={option.value}>{option.label}</option>
                    {/each}
                </select>
            </div>
            <div class="step-count-input">
                <label for="step-count">Steps:</label>
                <input
                    id="step-count"
                    type="number"
                    min="0"
                    max="1000"
                    value={stepCount}
                    oninput={handleStepCountChange}
                />
            </div>
            <div class="seed-input">
                <label for="seed">Seed:</label>
                <input
                    id="seed"
                    type="number"
                    min="-1"
                    max="2147483647"
                    value={seed}
                    oninput={handleSeedChange}
                    placeholder="-1 for random"
                />
            </div>
        {/if}
        <Selector
            service={modelSelectorService}
            pipelineType={pipelineType}
            placeholder={{ label: "Model:", id: "model" }}
            onModelSelect={handleModelSelect}
        />
    </div>
</div>

<style>
    .control-bar {
        display: grid;
        grid-template-columns: 1fr auto 1fr;
        align-items: center;
        padding: 1rem;
        background-color: #34495e;
        border-bottom: 2px solid #2c3e50;
        gap: 1rem;
    }

    .control-bar__label {
        text-align: left;
        color: white;
        margin: 0;
    }

    .control-bar :global(button) {
        justify-self: center;
    }

    .control-bar__selector {
        justify-self: end;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .dimension-selector {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .dimension-selector label {
        color: white;
        font-size: 0.875rem;
        font-weight: 500;
    }

    .dimension-selector select {
        padding: 0.4rem;
        border: 1px solid #ccc;
        border-radius: 4px;
        font-size: 0.875rem;
        background-color: white;
    }

    .dimension-selector select:focus {
        outline: none;
        border-color: #4a90e2;
        box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.1);
    }

    .step-count-input {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .step-count-input label {
        color: white;
        font-size: 0.875rem;
        font-weight: 500;
    }

    .step-count-input input {
        width: 70px;
        padding: 0.4rem;
        border: 1px solid #ccc;
        border-radius: 4px;
        font-size: 0.875rem;
    }

    .step-count-input input:focus {
        outline: none;
        border-color: #4a90e2;
        box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.1);
    }

    .seed-input {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .seed-input label {
        color: white;
        font-size: 0.875rem;
        font-weight: 500;
    }

    .seed-input input {
        width: 100px;
        padding: 0.4rem;
        border: 1px solid #ccc;
        border-radius: 4px;
        font-size: 0.875rem;
    }

    .seed-input input:focus {
        outline: none;
        border-color: #4a90e2;
        box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.1);
    }
</style>