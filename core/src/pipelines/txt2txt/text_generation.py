# Copyright (C) 2024-2025 FableYard
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Generic Text-to-Text Generation Pipeline

Supports any HuggingFace transformers-compatible causal LM model including:
- Qwen3
- Llama 3.1
- Mistral
- Gemma 2
"""

from gc import collect
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.logger import info


class TextGenerationPipeline:
    """
    Generic text generation pipeline using HuggingFace transformers.

    Uses device_map="auto" for CPU offloading, allowing large models
    to run on systems with limited GPU VRAM.
    """

    def __init__(
        self,
        model_path: Path,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.6,
        top_k: int = 20,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        seed: int = -1,
        chat_template_kwargs: Optional[dict] = None,
    ):
        """
        Initialize the text generation pipeline.

        Args:
            model_path: Path to the model directory
            prompt: Input text prompt for generation
            max_new_tokens: Maximum number of tokens to generate (default: 512)
            temperature: Sampling temperature (default: 0.6)
            top_k: Top-k sampling parameter (default: 20)
            top_p: Top-p (nucleus) sampling parameter (default: 0.95)
            repetition_penalty: Penalty for repeating tokens (default: 1.1, >1 discourages repeats)
            seed: Random seed for reproducibility (-1 for random)
            chat_template_kwargs: Model-specific kwargs for apply_chat_template
        """
        self.model_path = Path(model_path)
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.seed = seed
        self.chat_template_kwargs = chat_template_kwargs or {}

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        info(f"TextGenerationPipeline initialized - device: {self.device}")
        info(f"Model path: {self.model_path}")

    def execute(self) -> str:
        """
        Execute the text generation pipeline.

        Returns:
            Generated text string (excluding the input prompt)
        """
        info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        info("Loading model with device_map='auto' for CPU offloading...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        # Set seed for reproducibility
        if self.seed >= 0:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
            info(f"Using seed: {self.seed}")
        else:
            import random
            random_seed = random.randint(0, 2147483647)
            torch.manual_seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_seed)
            info(f"Generated random seed: {random_seed}")

        # Format as chat message for proper Q&A behavior (if chat template available)
        if tokenizer.chat_template is not None:
            info("Formatting prompt with chat template...")
            messages = [
                {"role": "user", "content": self.prompt}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **self.chat_template_kwargs
            )
        else:
            # Fallback to raw text for base models without chat template
            info("No chat template available, using raw prompt...")
            text = self.prompt

        info("Tokenizing input...")
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        input_length = inputs.input_ids.shape[1]
        info(f"Input length: {input_length} tokens")

        info(f"Generating up to {self.max_new_tokens} new tokens...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode only the generated tokens (exclude input)
        generated_tokens = outputs[0][input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        info(f"Generated {len(generated_tokens)} tokens")

        # Cleanup
        info("Cleaning up model from memory...")
        del model
        del tokenizer
        del inputs
        del outputs
        collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        info("TextGenerationPipeline execution complete")
        return generated_text
