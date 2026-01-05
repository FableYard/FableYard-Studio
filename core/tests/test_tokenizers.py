"""
Test script for CLIP and T5 tokenizers.

Tests basic functionality including encoding, decoding, batch operations,
and special token handling.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from text.tokenizers import CLIPTokenizer, T5Tokenizer


def test_clip_tokenizer():
    """Test CLIP tokenizer with local model files."""
    print("=" * 60)
    print("Testing CLIP Tokenizer")
    print("=" * 60)

    # Initialize tokenizer
    model_path = "models/FLUX-1-dev-0-30-0/tokenizer"
    tokenizer = CLIPTokenizer(model_path)

    # Test basic properties
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Max length: {tokenizer.max_length}")
    print(f"BOS token ID: {tokenizer.bos_token_id}")
    print(f"EOS token ID: {tokenizer.eos_token_id}")
    print(f"PAD token ID: {tokenizer.pad_token_id}")
    print()

    # Test encoding
    test_text = "a photo of a cat"
    print(f"Test text: '{test_text}'")
    tokens = tokenizer.encode(test_text, add_special_tokens=True)
    print(f"Encoded tokens: {tokens}")
    print(f"Token count: {len(tokens)}")

    # Test decoding
    decoded = tokenizer.decode(tokens)
    print(f"Decoded text: '{decoded}'")
    print()

    # Test batch encoding
    batch_texts = ["a photo of a cat", "a beautiful sunset", "mountains and trees"]
    print(f"Batch texts: {batch_texts}")
    batch_tokens = tokenizer.batch_encode(batch_texts, padding=True)
    print(f"Batch encoded shapes: {[len(t) for t in batch_tokens]}")
    print(f"First sequence: {batch_tokens[0]}")

    # Test batch decoding
    batch_decoded = tokenizer.batch_decode(batch_tokens)
    print(f"Batch decoded: {batch_decoded}")
    print()

    # Test truncation
    long_text = "a photo of " * 20
    print(f"Long text (first 50 chars): '{long_text[:50]}...'")
    long_tokens = tokenizer.encode(long_text, truncation=True)
    print(f"Truncated token count: {len(long_tokens)} (max: {tokenizer.max_length})")
    print()

    print("[PASS] CLIP Tokenizer tests passed!\n")


def test_t5_tokenizer():
    """Test T5 tokenizer with local model files."""
    print("=" * 60)
    print("Testing T5 Tokenizer")
    print("=" * 60)

    # Initialize tokenizer
    model_path = "models/FLUX-1-dev-0-30-0/tokenizer_2/spiece.model"
    tokenizer = T5Tokenizer(model_path, extra_ids=100, max_length=512)

    # Test basic properties
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Max length: {tokenizer.max_length}")
    print(f"BOS token ID: {tokenizer.bos_token_id}")
    print(f"EOS token ID: {tokenizer.eos_token_id}")
    print(f"PAD token ID: {tokenizer.pad_token_id}")
    print(f"UNK token ID: {tokenizer.unk_token_id}")
    print()

    # Test encoding
    test_text = "a photo of a cat"
    print(f"Test text: '{test_text}'")
    tokens = tokenizer.encode(test_text, add_special_tokens=True)
    print(f"Encoded tokens: {tokens}")
    print(f"Token count: {len(tokens)}")

    # Test decoding
    decoded = tokenizer.decode(tokens)
    print(f"Decoded text: '{decoded}'")
    print()

    # Test batch encoding
    batch_texts = ["a photo of a cat", "a beautiful sunset", "mountains and trees"]
    print(f"Batch texts: {batch_texts}")
    batch_tokens = tokenizer.batch_encode(batch_texts, padding=True, max_length=20)
    print(f"Batch encoded shapes: {[len(t) for t in batch_tokens]}")
    print(f"First sequence: {batch_tokens[0]}")

    # Test batch decoding
    batch_decoded = tokenizer.batch_decode(batch_tokens)
    print(f"Batch decoded: {batch_decoded}")
    print()

    # Test extra_id tokens
    print("Testing extra_id tokens:")
    for i in [0, 5, 99]:
        extra_id = tokenizer.get_extra_id(i)
        print(f"  <extra_id_{i}> = {extra_id}")
    print()

    # Test truncation
    long_text = "a photo of " * 100
    print(f"Long text (first 50 chars): '{long_text[:50]}...'")
    long_tokens = tokenizer.encode(long_text, truncation=True, max_length=50)
    print(f"Truncated token count: {len(long_tokens)} (max: 50)")
    print()

    print("[PASS] T5 Tokenizer tests passed!\n")


def compare_encodings():
    """Compare CLIP and T5 tokenizations of the same text."""
    print("=" * 60)
    print("Comparing CLIP vs T5 Tokenization")
    print("=" * 60)

    clip_tok = CLIPTokenizer("models/FLUX-1-dev-0-30-0/tokenizer")
    t5_tok = T5Tokenizer("models/FLUX-1-dev-0-30-0/tokenizer_2/spiece.model")

    test_texts = [
        "a photo of a cat",
        "a beautiful landscape with mountains",
        "cyberpunk city at night, neon lights"
    ]

    for text in test_texts:
        print(f"\nText: '{text}'")
        clip_tokens = clip_tok.encode(text, add_special_tokens=False)
        t5_tokens = t5_tok.encode(text, add_special_tokens=False)
        print(f"  CLIP: {len(clip_tokens)} tokens - {clip_tokens[:10]}{'...' if len(clip_tokens) > 10 else ''}")
        print(f"  T5:   {len(t5_tokens)} tokens - {t5_tokens[:10]}{'...' if len(t5_tokens) > 10 else ''}")

    print()


if __name__ == "__main__":
    try:
        test_clip_tokenizer()
        test_t5_tokenizer()
        compare_encodings()
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
