"""
Simple round trip test for the tokenizer using pytest.
"""

import pytest
from .tokenizer import Tokenizer



@pytest.mark.parametrize("text,target_length", [
    ("0123456789", 0),          # All digits
    ("[123_456]", 3),           # Digits with special chars
    ("0_1_2_3", 40),             # Alternating digits and underscores
    ("[0]", 0),                 # Single digit in brackets
    ("_", 1),                   # Single underscore
    ("9876543210", 20),          # Reverse digits
    ("[_]", -1),                 # Brackets with underscore
    ("012_345[678]9", 30),       # Mixed format
])
def test_tokenizer_round_trip(text, target_length):
    """Test that tokenizer can encode text and decode it back to original text."""
    tokenizer = Tokenizer()
    
    # Encode: text -> list[int]
    token_ids, mask = tokenizer.encode(text, target_length=target_length)
    
    # Decode: list[int] -> text
    decoded_text = tokenizer.decode(token_ids)
    
    # Assert round trip is successful
    assert decoded_text == text, f"Round trip failed: '{text}' -> {token_ids} -> '{decoded_text}'"
