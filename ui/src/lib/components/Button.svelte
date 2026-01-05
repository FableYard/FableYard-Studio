<script lang="ts">
    interface ButtonProps {
        label: string
        onClick: () => void
        class?: string
    }

    let {
        label,
        onClick,
        class: className = '',
    }: ButtonProps = $props();
</script>

<button class="button {className}" onclick={onClick}>
    {label}
</button>

<style>
    .button {
        text-align: center;
        padding: 0.75rem 1rem;
        background-color: #3498db;
        color: white;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 0.875rem;
        transition: background-color 0.2s;
    }

    .button:hover {
        background-color: #2980b9;
    }

    .button:active {
        background-color: #1f6a99;
    }
</style>