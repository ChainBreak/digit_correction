"""
Simple round trip test for the tokenizer using pytest.
"""

import pytest
from .tokenizer import Tokenizer



@pytest.mark.parametrize("text,target_length", [
    ("0123456789", 11),          # All digits
    ("0123456789", 20),          # All digits
    ("1,2", 4),           # Digits with special chars
    ("1", 2),           # Digits with special chars
    ("2", 4),           # Digits with special chars

])
def test_tokenizer_round_trip(text, target_length):
    """Test that tokenizer can encode text and decode it back to original text."""
    tokenizer = Tokenizer()
    
    # Encode: text -> list[int]
    input_ids, target_ids, mask = tokenizer.encode(text, target_length=target_length)

    assert len(input_ids) == len(target_ids) == len(mask), f"Lengths of input_ids, target_ids, and mask are not equal"
    assert len(input_ids) == target_length, f"Length of input_ids is not equal to target_length"

    complete_ids = input_ids[0:1] + target_ids

    # Decode: list[int] -> text
    decoded_text = tokenizer.decode(complete_ids)
    
    # Assert round trip is successful
    assert decoded_text == text, f"Round trip failed: '{text}' -> {complete_ids} -> '{decoded_text}'"
