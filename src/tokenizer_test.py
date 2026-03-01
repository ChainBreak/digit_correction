"""
Simple round trip test for the tokenizer using pytest.
"""

import pytest
from .tokenizer import Tokenizer

@pytest.mark.parametrize("original_text", [
    "0123456789",  # All digits
    "0123456789", # All digits
    "1,2",         # Digits with special chars
    "1",           # Digits with special chars
    "2",           # Digits with special chars

])
def test_tokenizer_round_trip(original_text):
    """Test that tokenizer can encode text and decode it back to original text."""
    tokenizer = Tokenizer()
    
    # Encode: text -> list[int]
    token_ids = tokenizer.encode(original_text)

    final_text = tokenizer.decode(token_ids)
    assert final_text == original_text, f"Decoded text is not equal to original text"


tokenizer = Tokenizer()
s = tokenizer.sos_token
e = tokenizer.eos_token
p = tokenizer.pad_token
@pytest.mark.parametrize("token_ids, target_string", [
    ([7,p,p,s,1, 2, 3,e,p,8], "123"),
])
def test_tokenizer_decode(token_ids, target_string):
    decoded_string = tokenizer.decode(token_ids)
    assert decoded_string == target_string, f"Decoded string is not equal to target string"