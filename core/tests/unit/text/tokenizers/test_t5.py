"""
Unit tests for T5 tokenizer.

Tests the T5Tokenizer wrapper implementation using SentencePiece
with local model files.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from text.tokenizers import T5Tokenizer


def get_tokenizer():
    """Create T5 tokenizer instance for testing."""
    model_path = "models/FLUX-1-dev-0-30-0/tokenizer_2/spiece.model"
    return T5Tokenizer(model_path, extra_ids=100, max_length=512)


def test_encode__with_special_tokens_adds_eos():
    """Test encoding with special tokens adds EOS token."""
    # Arrange
    tokenizer = get_tokenizer()
    text = "a photo of a cat"

    # Act
    without_special = tokenizer.encode(text, add_special_tokens=False)
    with_special = tokenizer.encode(text, add_special_tokens=True)

    # Assert
    assert len(with_special) == len(without_special) + 1
    assert with_special[-1] == tokenizer.eos_token_id
    assert tokenizer.eos_token_id == 1  # Verify property works


def test_encode_decode__preserves_text():
    """Test that encoding then decoding preserves original text."""
    # Arrange
    tokenizer = get_tokenizer()
    original = "a beautiful landscape with mountains and trees"

    # Act
    encoded = tokenizer.encode(original, add_special_tokens=False)
    decoded = tokenizer.decode(encoded, skip_special_tokens=True)

    # Assert
    assert decoded == original
    assert isinstance(encoded, list)
    assert all(isinstance(token_id, int) for token_id in encoded)


def test_encode__with_truncation_and_padding():
    """Test that truncation and padding work correctly."""
    # Arrange
    tokenizer = get_tokenizer()
    long_text = "a photo of " * 100
    short_text = "cat"
    max_length = 20

    # Act
    truncated = tokenizer.encode(long_text, max_length=max_length, truncation=True, padding=False)
    padded = tokenizer.encode(short_text, max_length=max_length, padding=True, add_special_tokens=True)

    # Assert
    assert len(truncated) <= max_length
    assert len(padded) == max_length
    assert tokenizer.max_length == 512  # Verify property works
    assert tokenizer.pad_token_id == 0  # Verify property works
    # Check that padding was actually added
    pad_count = sum(1 for tid in padded if tid == tokenizer.pad_token_id)
    assert pad_count > 0


def test_decode__skip_special_tokens_filters_pad_and_eos():
    """Test that skip_special_tokens removes EOS and PAD tokens."""
    # Arrange
    tokenizer = get_tokenizer()
    text = "cat"
    token_ids = tokenizer.encode(text, max_length=10, padding=True, add_special_tokens=True)

    # Act
    result = tokenizer.decode(token_ids, skip_special_tokens=True)

    # Assert
    assert result == text
    assert "</s>" not in result


def test_batch_encode_decode__preserves_texts():
    """Test that batch encode/decode preserves all texts."""
    # Arrange
    tokenizer = get_tokenizer()
    originals = [
        "a photo of a cat",
        "cyberpunk city at night",
        "mountains and sunset"
    ]

    # Act
    encoded = tokenizer.batch_encode(originals, padding=False, add_special_tokens=False)
    decoded = tokenizer.batch_decode(encoded, skip_special_tokens=True)

    # Assert
    assert decoded == originals
    assert len(encoded) == 3
    assert all(isinstance(tokens, list) for tokens in encoded)


def test_get_extra_id__returns_correct_values():
    """Test get_extra_id returns correct token IDs for T5 special tokens."""
    # Arrange
    tokenizer = get_tokenizer()

    # Act
    extra_id_0 = tokenizer.get_extra_id(0)
    extra_id_5 = tokenizer.get_extra_id(5)
    extra_id_99 = tokenizer.get_extra_id(99)

    # Assert
    assert extra_id_0 == 31999
    assert extra_id_5 == 31994
    assert extra_id_99 == 31900
    assert tokenizer.vocab_size == 32000  # Verify property works


def test_get_extra_id__exceeds_max_raises_error():
    """Test get_extra_id raises error for invalid extra ID."""
    # Arrange
    tokenizer = get_tokenizer()
    invalid_id = 100

    # Act & Assert
    try:
        tokenizer.get_extra_id(invalid_id)
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        assert "Extra ID 100 exceeds maximum 99" in str(e)


if __name__ == "__main__":
    # Run all tests
    test_encode__with_special_tokens_adds_eos()
    print("[PASS] test_encode__with_special_tokens_adds_eos")

    test_encode_decode__preserves_text()
    print("[PASS] test_encode_decode__preserves_text")

    test_encode__with_truncation_and_padding()
    print("[PASS] test_encode__with_truncation_and_padding")

    test_decode__skip_special_tokens_filters_pad_and_eos()
    print("[PASS] test_decode__skip_special_tokens_filters_pad_and_eos")

    test_batch_encode_decode__preserves_texts()
    print("[PASS] test_batch_encode_decode__preserves_texts")

    test_get_extra_id__returns_correct_values()
    print("[PASS] test_get_extra_id__returns_correct_values")

    test_get_extra_id__exceeds_max_raises_error()
    print("[PASS] test_get_extra_id__exceeds_max_raises_error")

    print("\nAll T5 tokenizer tests passed!")
