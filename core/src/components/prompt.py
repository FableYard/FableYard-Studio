# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

"""Prompt class for storing text encoder outputs."""

import json
import re
import random
from pathlib import Path
from typing import Optional


class Prompt:
    """
    Container for text encoder prompt embeddings with wildcard support.

    Stores CLIP and/or T5 prompt embeddings for conditioning image generation.
    Only includes attributes that are not None.

    Supports wildcard text for dynamic prompting:
    - `/colors/` - Random selection from wildcards/colors.json
    - `/colors(+red)/` - Include additional values
    - `/colors(-blue)/` - Exclude specific values
    - `[red, green, blue]` - In-place random selection

    Args:
        clip_prompt: Optional CLIP text prompt string (wildcards auto-processed)
        t5_prompt: Optional T5 text prompt string (wildcards auto-processed)
        wildcards_dir: Path to wildcards directory (defaults to 'wildcards/')
    """

    def __init__(
        self,
        clip_prompt: Optional[str] = None,
        t5_prompt: Optional[str] = None,
        wildcards_dir: str = "wildcards",
    ):
        self.wildcards_dir = Path(wildcards_dir)

        # Process prompts
        if clip_prompt is not None:
            self.clip_prompt = self._process_prompt(clip_prompt)

        if t5_prompt is not None:
            self.t5_prompt = self._process_prompt(t5_prompt)

    def _process_prompt(self, text: str) -> str:
        """
        Process wildcards in a prompt string.

        Args:
            text: Input prompt text with wildcard patterns

        Returns:
            Processed text with wildcards replaced
        """
        def replace_pattern(match):
            pattern = match.group(0)

            # Handle in-place lists: [value1, value2, value3]
            if pattern.startswith('['):
                values = [v.strip() for v in pattern[1:-1].split(',')]
                return random.choice(values) if values else pattern

            # Handle wildcard references: /name/ or /name(modifiers)/
            wildcard_match = re.match(r'/(\w+)((?:\([^)]+\))?)/+', pattern)
            if wildcard_match:
                return self._pick_wildcard(wildcard_match.group(1), wildcard_match.group(2))

            return pattern

        # Replace in-place lists: [value1, value2]
        text = re.sub(r'\[[^\]]+\]', replace_pattern, text)

        # Replace wildcard references: /name/ or /name(modifiers)/
        text = re.sub(r'/\w+(?:\([^)]+\))?/', replace_pattern, text)

        return text

    def _load_wildcard(self, name: str) -> list[str]:
        """
        Load wildcard values from JSON file.

        Args:
            name: Wildcard name (e.g., 'colors')

        Returns:
            List of values, empty list if file not found or invalid
        """
        wildcard_path = self.wildcards_dir / f"{name}.json"

        if not wildcard_path.exists():
            return []

        try:
            with open(wildcard_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("values", [])
        except (json.JSONDecodeError, KeyError):
            return []

    def _pick_wildcard(self, name: str, modifiers: str = "") -> str:
        """
        Pick a random value from a wildcard with optional modifiers.

        Args:
            name: Wildcard name (e.g., 'colors')
            modifiers: Optional modifiers string (e.g., '(+red,-blue)')

        Returns:
            Randomly selected value
        """
        values = self._load_wildcard(name).copy()

        if not values:
            return f"/{name}{modifiers}/"  # Return original if not found

        # Process modifiers
        if modifiers:
            # Handle additions: (+value1,value2)
            add_match = re.search(r'\(\+([^)]+)\)', modifiers)
            if add_match:
                additions = [v.strip() for v in add_match.group(1).split(',')]
                values.extend(additions)

            # Handle exclusions: (-value1,value2)
            exclude_match = re.search(r'\(-([^)]+)\)', modifiers)
            if exclude_match:
                exclusions = [v.strip() for v in exclude_match.group(1).split(',')]
                values = [v for v in values if v not in exclusions]

        return random.choice(values) if values else f"/{name}{modifiers}/"

    def __repr__(self) -> str:
        """String representation of the Prompt object."""
        attrs = []
        if hasattr(self, "clip_prompt"):
            attrs.append("clip_prompt")
        if hasattr(self, "t5_prompt"):
            attrs.append("t5_prompt")

        return f"Prompt({', '.join(attrs)})"