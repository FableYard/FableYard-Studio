<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { mediaPanelService, type OutputImage } from '$lib/services/MediaPanelService';
	import { webSocketService } from '$lib/services/WebSocketService';

	let mediaItems = $state<Array<{ id: number; url: string; name: string }>>([]);

	onMount(async () => {
		// Load existing images
		const outputs = await mediaPanelService.getOutputs();
		mediaItems = outputs.map((output: OutputImage, index: number) => ({
			id: index,
			url: mediaPanelService.getImageUrl(output.filename),
			name: output.filename
		}));

		// Listen for new completed images
		webSocketService.subscribe((event: any) => {
			// Check for completed media_generation tasks with image_url
			if (event.task_type === 'media_generation' && event.status === 'completed' && event.result?.image_url) {
				const imageUrl = event.result.image_url;
				const filename = imageUrl.split('/').pop() || 'unknown.png';

				// Add new image to the top of the list
				mediaItems = [
					{
						id: Date.now(),
						url: imageUrl,
						name: filename
					},
					...mediaItems
				];
			}
		});
	});

	onDestroy(() => {
		// Cleanup if needed
	});
</script>

<div id="media-panel" class="panel">
	<h3>Media</h3>
	<div class="media-content">
		{#each mediaItems as item (item.id)}
			<div class="media-item">
				<img src={item.url} alt={item.name} />
				<span class="media-name">{item.name}</span>
			</div>
		{/each}
	</div>
</div>

<style>
	#media-panel {
		display: flex;
		flex-direction: column;
		background-color: #34495e;
		min-width: 300px;
		max-width: 300px;
		padding: 1rem;
		overflow-y: auto;
	}

	h3 {
		color: white;
		margin: 0 0 1rem 0;
		font-size: 1.125rem;
		text-align: center;
	}

	.media-content {
		flex: 1;
		background-color: #2c3e50;
		border-radius: 4px;
		padding: 1rem;
		display: flex;
		flex-direction: column;
		gap: 0.75rem;
		color: #95a5a6;
	}

	.media-item {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
		background-color: #34495e;
		border-radius: 4px;
		padding: 0.5rem;
		transition: transform 0.2s;
	}

	.media-item:hover {
		transform: scale(1.02);
		cursor: pointer;
	}

	.media-item img {
		width: 100%;
		height: auto;
		border-radius: 4px;
		object-fit: cover;
		max-height: 200px;
	}

	.media-name {
		font-size: 0.875rem;
		color: #bdc3c7;
		text-align: center;
		word-break: break-all;
	}
</style>
