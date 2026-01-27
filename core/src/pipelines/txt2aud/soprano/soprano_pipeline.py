# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Soprano TTS Pipeline

Complete text-to-speech pipeline using Soprano model.
Uses custom components instead of soprano-tts package.
"""

from pathlib import Path
from typing import List, Optional, Union

import torch
from torch import Tensor

from pipelines.txt2aud.base_pipeline import BaseTTSPipeline
from components.tts.text_processor import TextProcessor
from components.tts.encoders.soprano_encoder import SopranoEncoder
from components.tts.vocoders.vocos import VocosDecoder
from components.tts.audio_saver import AudioSaver
from utils.logger import info


class SopranoPipeline(BaseTTSPipeline):
    """
    Soprano TTS Pipeline.

    Generates speech from text using:
    1. TextProcessor: Normalizes and formats text
    2. SopranoEncoder: Converts text to audio tokens (hidden states)
    3. VocosDecoder: Converts audio tokens to waveform
    4. AudioSaver: Saves output to WAV file

    Args:
        model_path: Path to Soprano model directory.
        prompt: Text to synthesize.
        seed: Random seed (-1 for random).
        device: Device to use (None for auto-detect).
        top_p: Nucleus sampling threshold.
        temperature: Sampling temperature.
        repetition_penalty: Penalty for repeated tokens.
        retries: Number of retry attempts for hallucination.
    """

    # Audio constants from Soprano
    TOKEN_SIZE = 2048  # Samples per audio token
    SAMPLE_RATE = 32000  # Output sample rate

    def __init__(
        self,
        model_path: Union[str, Path],
        prompt: str,
        seed: int = -1,
        device: Optional[str] = None,
        top_p: float = 0.95,
        temperature: float = 0.0,
        repetition_penalty: float = 1.2,
        retries: int = 0,
    ):
        super().__init__(model_path, prompt, seed, device)

        self.top_p = top_p
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.retries = retries

        info(f"SopranoPipeline initialized - device: {self.device}")

    def execute(self) -> str:
        """
        Execute TTS pipeline.

        Returns:
            Path to saved audio file.
        """
        # Set seed
        used_seed = self._setup_seed()
        if self.seed >= 0:
            info(f"Using seed: {used_seed}")
        else:
            info(f"Generated random seed: {used_seed}")

        # Initialize components
        info(f"Loading Soprano TTS from {self.model_path}...")

        # 1. Process text
        text_processor = TextProcessor()
        prompts, tracking = text_processor.process_batch([self.prompt])
        info(f"Processed text into {len(prompts)} sentence(s)")

        # 2. Load encoder and generate hidden states
        encoder = SopranoEncoder(
            model_path=self.model_path,
            device=self.device,
        )

        info(f"Generating audio tokens for: {self.prompt[:50]}...")
        results = encoder.encode_with_retry(
            prompts,
            retries=self.retries,
            top_p=self.top_p,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
        )

        # Extract hidden states
        hidden_states = [r['hidden_state'] for r in results]

        # Clean up encoder
        encoder.cleanup()
        del encoder

        # 3. Load decoder and convert to audio
        decoder = VocosDecoder.from_pretrained(
            model_path=self.model_path,
            device=self.device,
        )

        # Decode all sentences
        audio_segments = self._decode_batch(decoder, hidden_states, tracking)

        # Clean up decoder
        del decoder

        # 4. Concatenate audio segments
        audio = self._concatenate_audio(audio_segments, tracking)

        # 5. Save audio
        output_path = self._get_output_path("tts")
        saver = AudioSaver(
            output_dir=output_path.parent,
            sample_rate=self.SAMPLE_RATE,
        )

        # Remove .wav extension since AudioSaver adds it
        filename = output_path.stem
        saved_path = saver.save(audio, filename)
        info(f"Audio saved to: {saved_path}")

        # Cleanup
        self.cleanup()

        return str(saved_path)

    def _decode_batch(
        self,
        decoder: VocosDecoder,
        hidden_states: List[Tensor],
        tracking: List,
    ) -> List[Tensor]:
        """
        Decode hidden states to audio in batches.

        Processes sentences sorted by length for efficient batching.

        Args:
            decoder: VocosDecoder instance.
            hidden_states: List of hidden state tensors.
            tracking: List of (text_idx, sentence_idx) tuples.

        Returns:
            List of audio tensors in original order.
        """
        # Sort by length (longest first) for efficient batching
        combined = list(zip(hidden_states, tracking, range(len(hidden_states))))
        combined.sort(key=lambda x: -x[0].size(0))

        sorted_states, sorted_tracking, original_indices = zip(*combined)

        # Prepare output storage
        audio_outputs = [None] * len(hidden_states)

        # Process one at a time (batching could be added later)
        for idx, (hidden_state, track, orig_idx) in enumerate(zip(sorted_states, sorted_tracking, original_indices)):
            # Prepare input: (B, C, T)
            # hidden_state is (T, C), need (1, C, T)
            if hidden_state.dim() == 1:
                # Single token
                hidden_state = hidden_state.unsqueeze(0)

            x = hidden_state.unsqueeze(0).transpose(1, 2).to(self.device).to(torch.float32)

            # Decode
            with torch.no_grad():
                audio = decoder(x)

            # Trim to correct length
            # Output has extra samples from upsampling, trim based on token count
            num_tokens = hidden_state.size(0)
            expected_samples = (num_tokens - 1) * self.TOKEN_SIZE
            if expected_samples > 0:
                audio = audio[0, -expected_samples:]
            else:
                audio = audio[0]

            audio_outputs[orig_idx] = audio.cpu()

        return audio_outputs

    def _concatenate_audio(
        self,
        audio_segments: List[Tensor],
        tracking: List,
    ) -> Tensor:
        """
        Concatenate audio segments in correct order.

        Args:
            audio_segments: List of audio tensors.
            tracking: List of (text_idx, sentence_idx) tuples.

        Returns:
            Concatenated audio tensor.
        """
        # Group by text index
        num_texts = max(t[0] for t in tracking) + 1
        grouped = [[] for _ in range(num_texts)]

        for audio, (text_idx, sentence_idx) in zip(audio_segments, tracking):
            # Ensure we have enough slots
            while len(grouped[text_idx]) <= sentence_idx:
                grouped[text_idx].append(None)
            grouped[text_idx][sentence_idx] = audio

        # Concatenate each text's sentences
        result_audios = []
        for text_segments in grouped:
            # Filter out None values (shouldn't happen)
            valid_segments = [s for s in text_segments if s is not None]
            if valid_segments:
                result_audios.append(torch.cat(valid_segments))

        # Concatenate all texts
        if len(result_audios) == 1:
            return result_audios[0]
        return torch.cat(result_audios)


# Alias for backwards compatibility
SopranoTTSPipeline = SopranoPipeline
