
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
    ) -> tuple[list[int], list[int], list[int]]:
    
        token_ids = [self._vocab[c] for c in text]
        input_token_ids = [self._sos_token] + token_ids
        target_token_ids = token_ids + [self._eos_token]
        mask = [1] * len(input_token_ids)

        padding_needed = max(target_length - len(input_token_ids), 0)
        input_token_ids += [self._pad_token] * padding_needed
        target_token_ids += [self._pad_token] * padding_needed
        mask += [0] * padding_needed

        return input_token_ids, target_token_ids, mask
    
    def decode(
        self,
        token_ids: list[int],
        **kwargs
    ) -> str:

        text = "".join([self._reverse_vocab[i] for i in token_ids])
        text = text.replace("<PAD>", "")
        text = text.replace("<SOS>", "")
        text = text.replace("<EOS>", "")
        return text
    
    