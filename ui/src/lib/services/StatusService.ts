import type {
	StatusService,
	StatusEntry,
	QueuedStatus,
	ProcessingStatus,
	CompleteStatus,
	FailedStatus
} from '$lib/types/status-service';
import { webSocketService } from './WebSocketService';
import { writable } from 'svelte/store';

class StatusServiceImpl implements StatusService {
	private statusesStore = writable<StatusEntry[]>([]);
	private unsubscribe: (() => void) | null = null;

	constructor() {
		// Subscribe to WebSocket events
		this.unsubscribe = webSocketService.subscribe(this.handleEvent.bind(this));
	}

	// Expose the store for reactive subscriptions
	get statuses() {
		return this.statusesStore;
	}

	private handleEvent(event: any): void {
		const { task_id, task_type, error, result, payload } = event;

		// Map Redis events to status entries
		if (event.task_id && event.task_type === 'media_generation') {
			// Determine event type based on presence of error/result or explicit type
			if (error) {
				// task.failed
				console.error('Generation failed:', {
					taskId: task_id,
					error: error
				});
				const status: FailedStatus = {
					id: `${task_id}-failed`,
					runId: task_id,
					timestamp: new Date().toISOString(),
					type: 'failed',
					errorMessage: error
				};
				this.addStatus(status);
			} else if (result) {
				// task.completed
				console.log('Generation completed:', {
					taskId: task_id,
					outputPath: result?.output_path,
					duration: result?.duration
				});
				const status: CompleteStatus = {
					id: `${task_id}-complete`,
					runId: task_id,
					timestamp: new Date().toISOString(),
					type: 'complete',
					outputPath: result?.output_path || '/outputs/unknown',
					duration: result?.duration || 0
				};
				this.addStatus(status);
			} else if (payload) {
				// task.queued - has payload with request details
				console.log('Task queued:', {
					taskId: task_id,
					pipelineType: payload.pipelineType,
					model: payload.model
				});
				const status: QueuedStatus = {
					id: `${task_id}-queued`,
					runId: task_id,
					timestamp: new Date().toISOString(),
					type: 'queued',
					queuePosition: 1, // TODO: This needs to be refactored into a variable instead of hard-coded
					pipelineType: payload.pipelineType || 'Unknown',
					model: payload.model || 'unknown'
				};
				this.addStatus(status);
			} else {
				// task.processing - no error, result, or payload
				console.log('Task processing:', { taskId: task_id });
				const status: ProcessingStatus = {
					id: `${task_id}-processing`,
					runId: task_id,
					timestamp: new Date().toISOString(),
					type: 'processing',
					next: task_id
				};
				this.addStatus(status);
			}
		}
	}

	getStatuses(): StatusEntry[] {
		let currentStatuses: StatusEntry[] = [];
		this.statusesStore.subscribe(statuses => {
			currentStatuses = [...statuses];
		})();
		return currentStatuses;
	}

	addStatus(status: StatusEntry): void {
		// Add to beginning of array (newest first)
		this.statusesStore.update(statuses => [status, ...statuses]);
	}

	clearStatuses(): void {
		this.statusesStore.set([]);
	}

	destroy(): void {
		if (this.unsubscribe) {
			this.unsubscribe();
		}
	}
}

// Export singleton instance
export const statusService = new StatusServiceImpl();
