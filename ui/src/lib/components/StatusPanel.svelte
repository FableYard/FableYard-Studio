<script lang="ts">
	import { statusService } from '$lib/services/StatusService';
	import type {
		StatusEntry,
		QueuedStatus,
		ProcessingStatus,
		CompleteStatus,
		FailedStatus
	} from '$lib/types/status-service';

	// Subscribe to the reactive store
	let statuses = $state<StatusEntry[]>([]);

	// Reactively subscribe to status updates
	$effect(() => {
		const unsubscribe = statusService.statuses.subscribe(value => {
			statuses = value;
		});
		return unsubscribe;
	});

	// Helper functions for formatting
	function formatTimestamp(isoString: string): string {
		const date = new Date(isoString);
		return date.toLocaleString('en-US', {
			month: 'short',
			day: 'numeric',
			hour: '2-digit',
			minute: '2-digit',
			second: '2-digit'
		});
	}

	function formatDuration(ms: number): string {
		const seconds = Math.floor(ms / 1000);
		if (seconds < 60) return `${seconds}s`;

		const minutes = Math.floor(seconds / 60);
		const remainingSeconds = seconds % 60;
		if (minutes < 60) return `${minutes}m ${remainingSeconds}s`;

		const hours = Math.floor(minutes / 60);
		const remainingMinutes = minutes % 60;
		return `${hours}h ${remainingMinutes}m`;
	}

	function truncateWithTooltip(
		text: string,
		maxLength: number
	): { display: string; full: string } {
		if (text.length <= maxLength) {
			return { display: text, full: text };
		}
		return {
			display: text.substring(0, maxLength) + '...',
			full: text
		};
	}
</script>

<div id="status-panel" class="panel">
	<div class="status-container">
		{#each statuses as status (status.id)}
			<div class="status-card {status.type}">
				<div class="status-header">
					<span class="status-badge {status.type}">
						{status.type.toUpperCase()}
					</span>
					<span class="status-runid" title={status.runId}>
						{truncateWithTooltip(status.runId, 15).display}
					</span>
					<span class="status-timestamp">
						{formatTimestamp(status.timestamp)}
					</span>
				</div>
				<div class="status-body">
					{#if status.type === 'queued'}
						{@const queuedStatus = status as QueuedStatus}
						<div class="status-info">
							<span class="info-label">Position:</span>
							<span class="info-value">#{queuedStatus.queuePosition}</span>
						</div>
						<div class="status-info">
							<span class="info-label">Pipeline:</span>
							<span class="info-value">{queuedStatus.pipelineType}</span>
						</div>
						<div class="status-info">
							<span class="info-label">Model:</span>
							<span class="info-value" title={queuedStatus.model}>
								{truncateWithTooltip(queuedStatus.model, 30).display}
							</span>
						</div>
					{:else if status.type === 'processing'}
						{@const processingStatus = status as ProcessingStatus}
						<div class="status-info">
							<span class="info-label">Next Step:</span>
							<span class="info-value uuid" title={processingStatus.next}>
								{truncateWithTooltip(processingStatus.next, 20).display}
							</span>
						</div>
					{:else if status.type === 'complete'}
						{@const completeStatus = status as CompleteStatus}
						<div class="status-info">
							<span class="info-label">Output:</span>
							<span class="info-value path" title={completeStatus.outputPath}>
								{truncateWithTooltip(completeStatus.outputPath, 40).display}
							</span>
						</div>
						<div class="status-info">
							<span class="info-label">Duration:</span>
							<span class="info-value">{formatDuration(completeStatus.duration)}</span>
						</div>
					{:else if status.type === 'failed'}
						{@const failedStatus = status as FailedStatus}
						<div class="status-info error">
							<span class="info-label">Error:</span>
							<span class="info-value error-message" title={failedStatus.errorMessage}>
								{truncateWithTooltip(failedStatus.errorMessage, 60).display}
							</span>
						</div>
					{/if}
				</div>
			</div>
		{/each}
	</div>
</div>

<style>
	#status-panel {
		display: flex;
		flex-direction: column;
		background-color: #ecf0f1;
		border-top: 2px solid #34495e;
		padding: 0.5rem 1rem;
		height: 200px;
		overflow: hidden;
	}

	.status-container {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
		overflow-y: auto;
		padding: 0.25rem;
	}

	/* Status card base styles */
	.status-card {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
		padding: 0.75rem;
		border-radius: 4px;
		background-color: white;
		border-left: 4px solid;
		box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
		transition: box-shadow 0.2s ease;
	}

	.status-card:hover {
		box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
	}

	/* Color coding by status type */
	.status-card.queued {
		border-left-color: #3498db; /* Blue */
	}

	.status-card.processing {
		border-left-color: #f39c12; /* Yellow/Orange */
	}

	.status-card.complete {
		border-left-color: #27ae60; /* Green */
	}

	.status-card.failed {
		border-left-color: #e74c3c; /* Red */
	}

	/* Status header (badge, runId, timestamp) */
	.status-header {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		font-size: 0.875rem;
	}

	.status-badge {
		padding: 0.125rem 0.5rem;
		border-radius: 3px;
		font-weight: 600;
		font-size: 0.75rem;
		color: white;
	}

	.status-badge.queued {
		background-color: #3498db;
	}

	.status-badge.processing {
		background-color: #f39c12;
	}

	.status-badge.complete {
		background-color: #27ae60;
	}

	.status-badge.failed {
		background-color: #e74c3c;
	}

	.status-runid {
		font-family: 'Courier New', monospace;
		font-size: 0.8rem;
		color: #34495e;
		font-weight: 500;
	}

	.status-timestamp {
		margin-left: auto;
		color: #7f8c8d;
		font-size: 0.75rem;
	}

	/* Status body (details) */
	.status-body {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
	}

	.status-info {
		display: flex;
		gap: 0.5rem;
		font-size: 0.8rem;
	}

	.info-label {
		color: #7f8c8d;
		font-weight: 500;
		min-width: 70px;
	}

	.info-value {
		color: #2c3e50;
		flex: 1;
	}

	.info-value.uuid,
	.info-value.path {
		font-family: 'Courier New', monospace;
		font-size: 0.75rem;
	}

	.info-value.error-message {
		color: #e74c3c;
		font-weight: 500;
	}

	/* Scrollbar styling */
	.status-container::-webkit-scrollbar {
		width: 8px;
	}

	.status-container::-webkit-scrollbar-track {
		background: #ecf0f1;
		border-radius: 4px;
	}

	.status-container::-webkit-scrollbar-thumb {
		background: #bdc3c7;
		border-radius: 4px;
	}

	.status-container::-webkit-scrollbar-thumb:hover {
		background: #95a5a6;
	}
</style>
