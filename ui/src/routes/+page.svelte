<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import PipelinePanel from '$lib/components/PipelinePanel.svelte';
	import MediaPanel from '$lib/components/MediaPanel.svelte';
	import ControlBar from '$lib/components/ControlBar.svelte';
	import PromptPanel from '$lib/components/PromptPanel.svelte';
	import AdapterPanel from '$lib/components/AdapterPanel.svelte';
	import StatusPanel from '$lib/components/StatusPanel.svelte';
	import { modelPromptService } from '$lib/services/ModelPromptService';
	import { adapterSelectorService } from '$lib/services/AdapterSelectorService';
	import { webSocketService } from '$lib/services/WebSocketService';
	import { mediaGenerationService } from '$lib/services/MediaGenerationService';

	let selectedPipeline = $state('');
	let selectedModelFamily = $state('');
	let selectedModelName = $state('');
	let stepCount = $state(20);
	let imageDimensions = $state('512x512');
	let seed = $state(-1);
	let promptPanelRef: any;
	let adapterPanelRef: any;

	onMount(() => {
		webSocketService.connect();
	});

	onDestroy(() => {
		webSocketService.disconnect();
	});

	function handlePipelineSelect(pipelineType: string) {
		selectedPipeline = pipelineType;
		console.log('Selected pipeline:', pipelineType);
	}

	function handleModelSelect(modelFamily: string, modelName: string) {
		selectedModelFamily = modelFamily;
		selectedModelName = modelName;
		console.log('Model selected:', { family: selectedModelFamily, name: selectedModelName });
	}

	function handleStepCountChange(steps: number) {
		stepCount = steps;
		console.log('Step count changed:', steps);
	}

	function handleDimensionsChange(dimensions: string) {
		imageDimensions = dimensions;
		console.log('Dimensions changed:', dimensions);
	}

	function handleSeedChange(newSeed: number) {
		// Clamp seed value: -1 to 2147483647
		seed = Math.max(-1, Math.min(2147483647, newSeed));
		console.log('Seed changed:', seed);
	}

	async function handleGenerate() {
		if (!selectedPipeline || !selectedModelFamily || !selectedModelName) {
			console.error('❌ Cannot generate: Missing required fields', {
				selectedPipeline,
				selectedModelFamily,
				selectedModelName
			});
			return;
		}

		// Get prompts from PromptPanel
		const rawPrompts = promptPanelRef?.getPromptValues() || {};

		// Transform prompts from "EncoderName-promptType" format to API format
		// Example: {"CLIP-positive": "...", "T5-positive": "..."}
		// becomes: {"clip": {"positive": "..."}, "t5": {"positive": "..."}}
		const promptsDict: Record<string, Record<string, string>> = {};

		for (const [key, value] of Object.entries(rawPrompts)) {
			// Split "EncoderName-promptType" into parts
			const [encoderName, promptType] = key.split('-');
			if (encoderName && promptType && value) {
				const encoderKey = encoderName.toLowerCase();
				if (!promptsDict[encoderKey]) {
					promptsDict[encoderKey] = {};
				}
				promptsDict[encoderKey][promptType] = value as string;
			}
		}

		// Validate that we have at least one prompt
		if (Object.keys(promptsDict).length === 0) {
			console.error('❌ Cannot generate: No prompts provided', { rawPrompts });
			return;
		}

		// Get adapters from AdapterPanel (if any)
		const adapters = adapterPanelRef?.getAdapterValues();

		const model = `${selectedModelFamily}/${selectedModelName}`;

		// Parse dimensions (e.g., "512x768" -> width=512, height=768)
		const [width, height] = imageDimensions.split('x').map(Number);

		console.log('🚀 Starting generation request...', {
			pipelineType: selectedPipeline,
			model: model,
			prompts: promptsDict,
			adapters: adapters,
			stepCount,
			seed,
			width,
			height
		});

		try {
			const response = await mediaGenerationService.generateMedia({
				pipelineType: selectedPipeline,
				model: model,
				prompts: promptsDict,
				stepCount: stepCount,
				imageWidth: width,
				imageHeight: height,
				seed: seed,
				adapters: adapters
			});
			console.log('✅ Generation queued successfully!', response);
		} catch (error) {
			console.error('❌ Failed to queue generation:', error);
			if (error instanceof Error) {
				console.error('Error details:', {
					message: error.message,
					stack: error.stack
				});
			}
		}
	}
</script>

<div class="app-container">
	<header class="header">
		<h1>FableYard Studio</h1>
	</header>

	<main class="main">
		<PipelinePanel onPipelineSelect={handlePipelineSelect} />
		<div class="middle-section">
			<div class="content-area">
				<ControlBar
					pipelineType={selectedPipeline || 'Select a pipeline'}
					stepCount={stepCount}
					imageDimensions={imageDimensions}
					seed={seed}
					onModelSelect={handleModelSelect}
					onStepCountChange={handleStepCountChange}
					onDimensionsChange={handleDimensionsChange}
					onSeedChange={handleSeedChange}
					onGenerate={handleGenerate}
				/>
				{#if selectedPipeline && selectedModelName}
					<div class="panels-container">
						<PromptPanel
							bind:this={promptPanelRef}
							service={modelPromptService}
							pipelineType={selectedPipeline}
							modelFamily={selectedModelFamily}
							modelName={selectedModelName}
						/>
						{#if selectedPipeline === 'Text to Image'}
							<AdapterPanel bind:this={adapterPanelRef} service={adapterSelectorService} />
						{/if}
					</div>
				{/if}
			</div>
			<StatusPanel />
		</div>
		<MediaPanel />
	</main>

	<footer class="footer">
		<p>© FableYard Studio</p>
	</footer>
</div>

<style>
	:global(html, body) {
		margin: 0;
		padding: 0;
		overflow: hidden;
		height: 100%;
		width: 100%;
	}

	.app-container {
		display: flex;
		flex-direction: column;
		height: 100vh;
		width: 100vw;
		overflow: hidden;
	}

	.header {
		background-color: #2c3e50;
		color: white;
		padding: 1rem 2rem;
		border-bottom: 2px solid #34495e;
	}

	.header h1 {
		margin: 0;
		font-size: 1.5rem;
	}

	.main {
		flex: 1;
		background-color: #ecf0f1;
		overflow: hidden;
		display: flex;
	}

	.middle-section {
		flex: 1;
		display: flex;
		flex-direction: column;
		overflow: hidden;
	}

	.content-area {
		flex: 1;
		overflow-y: auto;
		display: flex;
		flex-direction: column;
	}

	.panels-container {
		display: flex;
		flex: 1;
		gap: 1rem;
		overflow: hidden;
	}

	.panels-container :global(#prompt-panel),
	.panels-container :global(#adapter-panel) {
		flex: 1;
		overflow-y: auto;
	}

	.footer {
		background-color: #2c3e50;
		color: white;
		padding: 0.75rem 2rem;
		border-top: 2px solid #34495e;
		text-align: center;
	}

	.footer p {
		margin: 0;
		font-size: 0.875rem;
	}
</style>
