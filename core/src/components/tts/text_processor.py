# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Text Processor for TTS

Handles text normalization, splitting, and formatting for TTS models.
Converts raw text to the format expected by Soprano encoder.
"""

from typing import List, Tuple

from components.tts.modules.normalizers.text_normalizer import TextNormalizer, TextSplitter


class TextProcessor:
    """
    Complete text processing pipeline for TTS.

    Handles:
    - Text normalization (numbers, abbreviations, symbols)
    - Sentence splitting with length constraints
    - Format wrapping for model input (Soprano format)
    - Batch processing with sentence tracking
    """

    def __init__(
        self,
        min_sentence_length: int = 30,
        max_sentence_length: int = 300,
    ):
        """
        Initialize text processor.

        Args:
            min_sentence_length: Minimum length before merging with adjacent sentences.
            max_sentence_length: Maximum sentence length before splitting.
        """
        self.min_length = min_sentence_length
        self.max_length = max_sentence_length
        self.normalizer = TextNormalizer()
        self.splitter = TextSplitter(desired_length=1, max_length=max_sentence_length)

    def process(self, text: str) -> List[str]:
        """
        Process a single text input.

        Args:
            text: Raw input text.

        Returns:
            List of formatted prompts ready for the model.
        """
        results, _ = self.process_batch([text])
        return results

    def process_batch(
        self,
        texts: List[str]
    ) -> Tuple[List[str], List[Tuple[int, int]]]:
        """
        Process multiple texts with sentence tracking.

        Args:
            texts: List of raw input texts.

        Returns:
            Tuple of:
            - List of formatted prompts
            - List of (text_index, sentence_index) for tracking
        """
        results = []

        for text_idx, text in enumerate(texts):
            text = text.strip()

            # Normalize text
            cleaned_text = self.normalizer.normalize(text)

            # Split into sentences
            sentences = self.splitter.split(cleaned_text)

            # Build sentence list with tracking
            processed = []
            for sentence in sentences:
                processed.append({
                    "text": sentence,
                    "text_idx": text_idx,
                })

            # Merge short sentences if needed
            if self.min_length > 0 and len(processed) > 1:
                merged = []
                i = 0
                while i < len(processed):
                    cur = processed[i]
                    if len(cur["text"]) < self.min_length:
                        # Try to merge with previous or next
                        if merged:
                            merged[-1]["text"] = (merged[-1]["text"] + " " + cur["text"]).strip()
                        elif i + 1 < len(processed):
                            processed[i + 1]["text"] = (cur["text"] + " " + processed[i + 1]["text"]).strip()
                        else:
                            merged.append(cur)
                    else:
                        merged.append(cur)
                    i += 1
                processed = merged

            # Assign sentence indices and format for model
            sentence_indices = {}
            for item in processed:
                tid = item['text_idx']
                if tid not in sentence_indices:
                    sentence_indices[tid] = 0

                # Soprano format: [STOP][TEXT]...[START]
                prompt = f"[STOP][TEXT]{item['text']}[START]"
                tracking = (tid, sentence_indices[tid])

                results.append((prompt, tracking))
                sentence_indices[tid] += 1

        # Separate prompts and tracking info
        prompts = [r[0] for r in results]
        tracking = [r[1] for r in results]

        return prompts, tracking

    def normalize_only(self, text: str) -> str:
        """
        Only normalize text without splitting or formatting.

        Useful for debugging or when you need just the cleaned text.

        Args:
            text: Raw input text.

        Returns:
            Normalized text.
        """
        return self.normalizer.normalize(text)
