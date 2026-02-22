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
    input_ids, target_ids, position_indices, mask = tokenizer.encode(text, target_length=target_length)

    assert len(input_ids) == len(target_ids) == len(position_indices) == len(mask) , f"Lengths of input_ids, target_ids, position_indices, and mask are not equal"
    assert len(input_ids) == target_length, f"Length of input_ids is not equal to target_length"


tokenizer = Tokenizer()
s = tokenizer._sos_token
e = tokenizer._eos_token
p = tokenizer.pad_token
@pytest.mark.parametrize("token_ids, target_string", [
    ([7,p,p,s,1, 2, 3,e,p,8], "123"),
])
def test_tokenizer_decode(token_ids, target_string):
    decoded_string = tokenizer.decode(token_ids)
    assert decoded_string == target_string, f"Decoded string is not equal to target string"