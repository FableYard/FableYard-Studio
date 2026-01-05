<script lang="ts">
    import type {LoRAService} from "$lib/types/lora-service";
    import {onMount} from "svelte";

    interface LoRAPanelProps {
        service: LoRAService
    }

    let {service}: LoRAPanelProps = $props();

    let loraCards = $state<number[]>([]);
    let availableLoRAs = $state<string[]>([]);
    let loading = $state(true);
    let error = $state<string | null>(null);

    onMount(async () => {
        const result = await service.getLoRAs();
        if (result.ok) {
            availableLoRAs = result.value;
        } else {
            error = result.error;
        }
        loading = false;
    });

    function addLoRACard() {
        loraCards = [...loraCards, Date.now()];
    }

    function removeLoRACard(cardId: number) {
        loraCards = loraCards.filter(id => id !== cardId);
    }
</script>

<div id="lora-panel" class="panel">
    {#each loraCards as cardId (cardId)}
        <div class="lora-row">
            <div class="lora-card">
                <div class="lora-card-header">LoRA</div>
                <div class="lora-card-body">
                    <label>
                        LoRA:
                        <select class="lora-selector" disabled={loading}>
                            <option value="">
                                {#if loading}
                                    Loading LoRAs...
                                {:else if error}
                                    Error loading LoRAs
                                {:else}
                                    Select a LoRA
                                {/if}
                            </option>
                            {#if !loading && !error}
                                {#each availableLoRAs as lora}
                                    <option value={lora}>{lora}</option>
                                {/each}
                            {/if}
                        </select>
                    </label>
                </div>
            </div>
            <button class="remove-lora-button" onclick={() => removeLoRACard(cardId)}>x</button>
        </div>
    {/each}
    <div class="lora-row">
        <button class="add-lora-button" onclick={addLoRACard}>+</button>
    </div>
</div>

<style>
    #lora-panel {
        display: flex;
        flex-direction: column;
        padding: 1rem;
    }

    .lora-row {
        display: flex;
        align-items: flex-end;
    }

    .add-lora-button {
        width: 32px;
        height: 32px;
        font-size: 20px;
        cursor: pointer;
    }

    .remove-lora-button {
        width: 32px;
        height: 32px;
        font-size: 20px;
        cursor: pointer;
        margin-left: 8px;
    }

    .lora-card {
        flex: 1;
        border: 1px solid #ccc;
        border-radius: 4px;
        padding: 8px;
    }

    .lora-card-header {
        font-weight: bold;
        margin-bottom: 8px;
    }

    .lora-card-body {
        display: flex;
        flex-direction: column;
    }

    .lora-card-body label {
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .lora-selector {
        flex: 1;
    }
</style>