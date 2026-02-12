
class Tokenizer():
    
    def __init__(self):
        super().__init__()
        chars = list("0123456789,")
        chars.extend([
            "<SOS>",
            "<EOS>",
            "<PAD>",
        ])
        self._vocab = { k:i for i, k in enumerate(chars)}
        self._reverse_vocab = {i:k for k, i in self._vocab.items()}

        self._pad_token = self._vocab["<PAD>"]
        self._sos_token = self._vocab["<SOS>"]
        self._eos_token = self._vocab["<EOS>"]
       
    def __len__(self):
        return len(self._vocab)
        
    def encode(
        self,
        text: str,
        target_length: int = 0,
        **kwargs
    ) -> tuple[list[int], list[int], list[int], list[int]]:
    
        token_ids = [self._vocab[c] for c in text]
        input_token_ids = [self._sos_token] + token_ids
        target_token_ids = token_ids + [self._eos_token]
        positive_mask = [1] * len(input_token_ids)
        position_indices = list(range(len(input_token_ids)))

        # Create paddings
        padding_needed = max(target_length - len(input_token_ids), 0)
        padding = [self._pad_token] * padding_needed
        negative_mask = [0] * padding_needed
        blank_position_indices = [0] * padding_needed

        # Pad the front of the lists
        input_token_ids = padding + input_token_ids
        target_token_ids = padding + target_token_ids
        mask = negative_mask + positive_mask
        position_indices = blank_position_indices + position_indices

        return input_token_ids, target_token_ids, position_indices, mask
    
    def decode(
        self,
        token_ids: list[int],
        **kwargs
    ) -> str:

        # Find the first index where the token id is equal to self._sos_token
        
        sos_index = token_ids.index(self._sos_token)
        eos_index = token_ids.index(self._eos_token)

        token_ids = token_ids[sos_index + 1:eos_index]

        text = "".join([self._reverse_vocab[i] for i in token_ids])
        
        return text
    
    