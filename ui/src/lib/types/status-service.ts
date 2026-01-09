// Status type discriminated union
export type StatusType = 'queued' | 'processing' | 'complete' | 'failed';

// Base status interface with common properties
interface BaseStatusEntry {
	id: string; // Unique identifier for the status entry
	runId: string; // Pipeline run ID
	timestamp: string; // ISO 8601 string for easy serialization
	type: StatusType; // Discriminator
}

// Specific status types using discriminated union pattern
export interface QueuedStatus extends BaseStatusEntry {
	type: 'queued';
	queuePosition: number;
	pipelineType: string;
	model: string;
}

export interface ProcessingStatus extends BaseStatusEntry {
	type: 'processing';
	next: string; // UUID of next step
}

export interface CompleteStatus extends BaseStatusEntry {
	type: 'complete';
	outputPath: string;
	duration: number; // Duration in milliseconds
}

export interface FailedStatus extends BaseStatusEntry {
	type: 'failed';
	errorMessage: string;
}

// Union type for all status entries
export type StatusEntry = QueuedStatus | ProcessingStatus | CompleteStatus | FailedStatus;

// Service interface
export interface StatusService {
	// Get all current status entries (returns array directly, not wrapped in Result)
	getStatuses(): StatusEntry[];

	// Add a new status entry (for future use when integrating with real pipeline)
	addStatus(status: StatusEntry): void;

	// Optional: Clear all statuses (for testing/reset)
	clearStatuses(): void;
}
