"""
Unit tests for CLIP tokenizer.

Tests the CLIPTokenizer wrapper implementation using ByteLevelBPE
with local model files.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from text.tokenizers import CLIPTokenizer


class TestCLIPTokenizer:
    def test_encoder(self):
        ...

    def test_decode(self):
        ...

    def test_batch_encode(self):
        ...

    def test_batch_decode(self):
        ...



def get_tokenizer():
    """Create CLIP tokenizer instance for testing."""
    model_path = "models/FLUX-1-dev-0-30-0/tokenizer"
    return CLIPTokenizer(model_path)


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


def test_encode__with_truncation_limits_to_max_length():
    """Test that truncation limits token count to max_length."""
    # Arrange
    tokenizer = get_tokenizer()
    long_text = "a photo of " * 50
    max_length = 20

    # Act
    result = tokenizer.encode(long_text, max_length=max_length, truncation=True)

    # Assert
    assert len(result) <= max_length
    assert tokenizer.max_length == 77  # Verify property works


def test_encode__with_padding_extends_to_max_length():
    """Test that padding extends sequence to max_length."""
    # Arrange
    tokenizer = get_tokenizer()
    short_text = "cat"
    max_length = 20

    # Act
    result = tokenizer.encode(short_text, max_length=max_length, padding=True)

    # Assert
    assert len(result) == max_length
    assert tokenizer.pad_token_id == 49407  # Verify property works


def test_decode__skip_special_tokens_filters_padding():
    """Test that skip_special_tokens removes pad tokens from output."""
    # Arrange
    tokenizer = get_tokenizer()
    text = "cat"
    token_ids = tokenizer.encode(text, max_length=10, padding=True)

    # Act
    result = tokenizer.decode(token_ids, skip_special_tokens=True)

    # Assert
    assert result == text
    assert "<|endoftext|>" not in result


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


if __name__ == "__main__":
    # Run all tests
    test_encode_decode__preserves_text()
    print("[PASS] test_encode_decode__preserves_text")

    test_encode__with_truncation_limits_to_max_length()
    print("[PASS] test_encode__with_truncation_limits_to_max_length")

    test_encode__with_padding_extends_to_max_length()
    print("[PASS] test_encode__with_padding_extends_to_max_length")

    test_decode__skip_special_tokens_filters_padding()
    print("[PASS] test_decode__skip_special_tokens_filters_padding")

    test_batch_encode_decode__preserves_texts()
    print("[PASS] test_batch_encode_decode__preserves_texts")

    print("\nAll CLIP tokenizer tests passed!")
