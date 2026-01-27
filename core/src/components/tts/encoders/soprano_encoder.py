# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Soprano Encoder

Text-to-audio-tokens encoder using the Soprano model architecture.
Uses a Qwen3-based causal language model to generate hidden states
that represent audio tokens.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer


class SopranoEncoder:
    """
    Soprano text-to-audio encoder.

    Generates audio token hidden states from text input using a
    Qwen3-based causal language model. The hidden states are then
    passed to the Vocos decoder to produce audio.

    Args:
        model_path: Path to the model directory.
        device: Device to run on ("cuda", "cpu", etc.).
    """

    # Constants
    RECEPTIVE_FIELD = 4  # Decoder receptive field in tokens
    TOKEN_SIZE = 2048    # Samples per audio token

    def __init__(
        self,
        model_path: Union[str, Path],
        device: Union[str, torch.device] = "cuda",
    ):
        self.model_path = Path(model_path)
        self.device = torch.device(device)

        # Determine dtype based on device
        if self.device.type == "cuda":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            dtype=self.dtype,
            device_map=str(self.device),
            local_files_only=True,
        )
        self.model.eval()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            local_files_only=True,
        )

        # Get special token IDs
        self.eos_token_id = self.model.config.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

    @torch.no_grad()
    def encode(
        self,
        prompts: List[str],
        top_p: float = 0.95,
        temperature: float = 0.001,
        repetition_penalty: float = 1.2,
        max_new_tokens: int = 512,
    ) -> List[Dict]:
        """
        Encode text prompts to audio token hidden states.

        Args:
            prompts: List of formatted text prompts.
            top_p: Nucleus sampling threshold.
            temperature: Sampling temperature (near 0 = greedy).
            repetition_penalty: Penalty for repeated tokens.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            List of dicts with 'hidden_state' and 'finish_reason' keys.
        """
        # Temperature must be non-zero
        if temperature <= 0.0:
            temperature = 0.001

        # Tokenize inputs
        inputs = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        # Generate with hidden states output
        outputs = self.model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.pad_token_id,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )

        # Extract hidden states for each prompt
        results = []
        for i in range(len(prompts)):
            seq = outputs.sequences[i]
            hidden_states = []

            num_output_tokens = len(outputs.hidden_states)
            for j in range(num_output_tokens):
                token = seq[j + seq.size(0) - num_output_tokens]
                # Skip EOS tokens
                if token != self.eos_token_id:
                    # Get last layer hidden state for this token
                    hidden_states.append(outputs.hidden_states[j][-1][i, -1, :])

            # Stack into tensor
            if hidden_states:
                last_hidden_state = torch.stack(hidden_states).squeeze()
            else:
                # Empty output
                last_hidden_state = torch.zeros(1, self.model.config.hidden_size, device=self.device)

            # Determine finish reason
            finish_reason = 'stop' if seq[-1].item() == self.eos_token_id else 'length'

            results.append({
                'finish_reason': finish_reason,
                'hidden_state': last_hidden_state,
            })

        return results

    def detect_hallucination(
        self,
        hidden_state: Tensor,
        diff_threshold: float = 300,
        max_runlength: int = 16,
    ) -> bool:
        """
        Detect potential hallucination in generated hidden states.

        Looks for long runs of similar sequences which indicate
        the model is stuck in a loop.

        Args:
            hidden_state: Hidden states tensor of shape (T, hidden_dim).
            diff_threshold: Minimum difference between sequences.
            max_runlength: Maximum allowed run of similar sequences.

        Returns:
            True if hallucination detected, False otherwise.
        """
        if len(hidden_state) <= max_runlength:
            return False

        runlength = 0
        for i in range(len(hidden_state) - 1):
            current = hidden_state[i]
            next_seq = hidden_state[i + 1]

            # Compute L1 difference
            diff = torch.abs(current - next_seq).sum()

            if diff < diff_threshold:
                runlength += 1
            elif runlength > 0:
                runlength -= 1

            if runlength > max_runlength:
                return True

        return False

    @torch.no_grad()
    def encode_with_retry(
        self,
        prompts: List[str],
        retries: int = 0,
        **kwargs
    ) -> List[Dict]:
        """
        Encode with automatic retry on hallucination detection.

        Args:
            prompts: List of formatted text prompts.
            retries: Number of retry attempts for hallucinated outputs.
            **kwargs: Additional arguments passed to encode().

        Returns:
            List of encoding results.
        """
        results = [None] * len(prompts)
        pending_indices = list(range(len(prompts)))
        tries_left = 1 + max(0, retries)

        while tries_left > 0 and pending_indices:
            current_prompts = [prompts[i] for i in pending_indices]
            responses = self.encode(current_prompts, **kwargs)

            bad_indices = []
            for idx, response in enumerate(responses):
                actual_idx = pending_indices[idx]
                results[actual_idx] = response

                # Check for incomplete generation
                if response['finish_reason'] != 'stop':
                    print(f"Warning: Sentence {actual_idx} did not complete generation.")

                # Check for hallucination if retries enabled
                if retries > 0 and self.detect_hallucination(response['hidden_state']):
                    print(f"Warning: Sentence {actual_idx} contains hallucination.")
                    bad_indices.append(actual_idx)

            if not bad_indices:
                break

            pending_indices = bad_indices
            tries_left -= 1
            if tries_left > 0:
                print(f"Retrying {len(pending_indices)} sentence(s)...")

        return results

    def cleanup(self):
        """Release model resources."""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'tokenizer'):
            del self.tokenizer

        import gc
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
