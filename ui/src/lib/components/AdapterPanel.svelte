<script lang="ts">
    import { onMount, onDestroy } from "svelte";
    import type { SelectorService } from "$lib/types/selector-service";
    import { modelSelectorService } from "$lib/services/ModelSelectorService";
    import Selector from "$lib/components/Selector.svelte";

    interface AdapterPanelProps {
        service: SelectorService
    }

    let { service }: AdapterPanelProps = $props();

    let adapterCards = $state<number[]>([]);
    let availableAdapters = $state<string[]>([]);
    let loading = $state(true);
    let error = $state<string | null>(null);
    let currentModelFamily = $state<string | null>(null);
    let adapterStrengths = $state<Map<number, number>>(new Map());
    let adapterSelections = $state<Map<number, string>>(new Map());
    let nextCardId = $state(1);

    let unsubscribe: () => void;

    onMount(() => {
        unsubscribe = modelSelectorService.selectedModelFamily.subscribe(family => {
            currentModelFamily = family;
            adapterCards = [];
            adapterStrengths.clear();
            adapterSelections.clear();
            nextCardId = 1;
            loadAdapters(family);
        });
    });

    onDestroy(() => unsubscribe());

    function addAdapterCard() {
        const newCardId = nextCardId;
        adapterCards = [...adapterCards, newCardId];
        adapterStrengths.set(newCardId, 1.0);
        nextCardId++;
    }

    function removeAdapterCard(cardId: number) {
        adapterCards = adapterCards.filter(id => id !== cardId);
        adapterStrengths.delete(cardId);
        adapterSelections.delete(cardId);
    }

    function updateStrength(cardId: number, value: number) {
        adapterStrengths.set(cardId, value);
    }

    function handleAdapterSelect(cardId: number, adapterName: string) {
        adapterSelections.set(cardId, adapterName);
    }

    // Export method to get adapter values in the correct format
    export function getAdapterValues(): Record<string, {path: string, strength: number}> | undefined {
        if (!currentModelFamily || adapterCards.length === 0) {
            return undefined;
        }

        const adapters: Record<string, {path: string, strength: number}> = {};

        for (const cardId of adapterCards) {
            const selectedAdapter = adapterSelections.get(cardId);
            const strength = adapterStrengths.get(cardId);

            // Only include cards with both valid selection and strength
            if (selectedAdapter && strength !== undefined) {
                // Remove .safetensors extension for the key
                const adapterKey = selectedAdapter.replace(/\.safetensors$/, '');

                // Construct relative path: modelFamily/filename
                const relativePath = `${currentModelFamily}/${selectedAdapter}`;

                adapters[adapterKey] = {
                    path: relativePath,
                    strength: strength
                };
            }
        }

        // Return undefined if no valid adapters, otherwise return the object
        return Object.keys(adapters).length > 0 ? adapters : undefined;
    }

    // function removeAllAdapterCards() {
    //     adapterCards = [];
    // }

    async function loadAdapters(modelFamily: string | null) {
        if (!modelFamily) {
            availableAdapters = [];
            return;
        }

        loading = true;
        const result = await service.getOptions(modelFamily)
        if (result.ok) {
            availableAdapters = result.value;
        } else {
            error = result.error;
        }
        loading = false;
    }
</script>

<div id="adapter-panel" class="panel">
    {#each adapterCards as cardId (cardId)}
        <div class="adapter-row">
            <div class="adapter-card">
                <div class="adapter-card-header">Adapter</div>
                <div class="adapter-card-body">
<!--                    <label>-->
<!--                        Adapter:-->
<!--                        <select class="adapter-selector" disabled={loading}>-->
<!--                            <option value="">-->
<!--                                {#if loading}-->
<!--                                    Loading adapters...-->
<!--                                {:else if error}-->
<!--                                    Error loading adapters: {error}-->
<!--                                {:else}-->
<!--                                    Select an adapter-->
<!--                                {/if}-->
<!--                            </option>-->
<!--                            {#if !loading && !error}-->
<!--                                {#each availableAdapters as adapter}-->
<!--                                    <option value={adapter}>{adapter}</option>-->
<!--                                {/each}-->
<!--                            {/if}-->
<!--                        </select>-->
<!--                    </label>-->
                    <Selector
                            service={service}
                            modelFamily={currentModelFamily}
                            placeholder={{ label: "Adapter", id: `adapter-${cardId}` }}
                            onAdapterSelect={(adapterName) => handleAdapterSelect(cardId, adapterName)}
                    />
                    <div class="strength-input">
                        <label for="strength-{cardId}">Strength:</label>
                        <input
                            id="strength-{cardId}"
                            type="number"
                            min="0"
                            max="2"
                            step="0.01"
                            value={adapterStrengths.get(cardId) ?? 1.0}
                            oninput={(e) => updateStrength(cardId, parseFloat((e.target as HTMLInputElement).value))}
                        />
                    </div>
                    <button class="remove-adapter-button" onclick={() => removeAdapterCard(cardId)}>X</button>
                </div>
            </div>
<!--            <button class="remove-adapter-button" onclick={() => removeAdapterCard(cardId)}>x</button>-->
        </div>
    {/each}
    <div class="adapter-row">
        <button class="add-adapter-button" onclick={addAdapterCard}>+</button>
    </div>
</div>

<style>
    #adapter-panel {
        display: flex;
        flex-direction: column;
        padding: 1rem;
    }

    .adapter-row {
        display: flex;
        align-items: flex-end;
    }

    .add-adapter-button {
        width: 32px;
        height: 32px;
        font-size: 20px;
        cursor: pointer;
    }

    .remove-adapter-button {
        width: 32px;
        height: 32px;
        font-size: 20px;
        cursor: pointer;
        margin-left: 8px;
    }

    .adapter-card {
        flex: 1;
        border: 1px solid #ccc;
        border-radius: 4px;
        padding: 8px;
    }

    .adapter-card-header {
        font-weight: bold;
        margin-bottom: 8px;
    }

    .adapter-card-body {
        display: flex;
        flex-direction: row;
        align-items: center;
        gap: 8px;
    }

    .adapter-card-body label {
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .strength-input {
        display: flex;
        align-items: center;
        gap: 4px;
    }

    .strength-input label {
        font-size: 0.875rem;
        font-weight: 500;
        white-space: nowrap;
    }

    .strength-input input {
        width: 70px;
        padding: 0.4rem;
        border: 1px solid #ccc;
        border-radius: 4px;
        font-size: 0.875rem;
    }

    .strength-input input:focus {
        outline: none;
        border-color: #4a90e2;
        box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.1);
    }

    .adapter-selector {
        flex: 1;
    }
</style>