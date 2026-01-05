<script lang="ts">
	import type { Snippet } from 'svelte';

	interface CardProps {
		title?: string;
		children: Snippet;
	}
	let { children, title }: CardProps = $props();
</script>

<article class="card">
	{#if title}
		<header class="card-header">
			<h3 class="card-title">{title}</h3>
		</header>
	{/if}

	<section class="card-body">
		{@render children()}
	</section>
</article>

<style>
	.card {
		background: white;
		border-radius: 8px;
		box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
		overflow: hidden;
		transition: box-shadow 0.2s ease-in-out;
	}

	.card-header {
		padding: 1.25rem 1.5rem;
		border-bottom: 1px solid #e5e7eb;
		background: #f9fafb;
	}

	.card-title {
		margin: 0;
		font-size: 1.25rem;
		font-weight: 600;
		color: #111827;
	}

	.card-body {
		padding: 1.5rem;
		color: #374151;
	}
</style>