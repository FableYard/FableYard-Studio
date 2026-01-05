<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import PipelinePanel from '$lib/components/PipelinePanel.svelte';
	import MediaPanel from '$lib/components/MediaPanel.svelte';
	import ControlBar from '$lib/components/ControlBar.svelte';
	import PromptPanel from '$lib/components/PromptPanel.svelte';
	import LoRAPanel from '$lib/components/LoRAPanel.svelte';
	import StatusPanel from '$lib/components/StatusPanel.svelte';
	import { modelPromptService } from '$lib/services/ModelPromptService';
	import { loraService } from '$lib/services/LoRAService';
	import { webSocketService } from '$lib/services/WebSocketService';
	import { mediaGenerationService } from '$lib/services/MediaGenerationService';

	let selectedPipeline = $state('');
	let selectedModelFamily = $state('');
	let selectedModelName = $state('');
	let stepCount = $state(20);
	let imageDimensions = $state('512x512');
	let promptPanelRef: any;

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

	function handleModelSelect(modelFullName: string) {
		const parts = modelFullName.split('/');
		if (parts.length === 2) {
			selectedModelFamily = parts[0];
			selectedModelName = parts[1];
			console.log('Model selected:', { family: selectedModelFamily, name: selectedModelName });
		}
	}

	function handleStepCountChange(steps: number) {
		stepCount = steps;
		console.log('Step count changed:', steps);
	}

	function handleDimensionsChange(dimensions: string) {
		imageDimensions = dimensions;
		console.log('Dimensions changed:', dimensions);
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

		const model = `${selectedModelFamily}/${selectedModelName}`;

		// Parse dimensions (e.g., "512x768" -> width=512, height=768)
		const [width, height] = imageDimensions.split('x').map(Number);

		console.log('🚀 Starting generation request...', {
			pipelineType: selectedPipeline,
			model: model,
			prompts: promptsDict,
			stepCount,
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
				imageHeight: height
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
					onModelSelect={handleModelSelect}
					onStepCountChange={handleStepCountChange}
					onDimensionsChange={handleDimensionsChange}
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
							<LoRAPanel service={loraService} />
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
	.panels-container :global(#lora-panel) {
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
