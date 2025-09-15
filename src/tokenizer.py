
class Tokenizer():
    
    def __init__(self):
        super().__init__()
        self._vocab = { k:i for i, k in enumerate(list("0123456789[,_]?"))}
        self._reverse_vocab = {i:k for k, i in self._vocab.items()}
        self._pad_char = "?"
        
    def encode(
        self,
        text: str,
        target_length: int = 0,
        **kwargs
    ) -> tuple[list[int], list[int]]:
    
        tokens = [self._vocab[c] for c in text]
        mask = [1] * len(tokens)

        padding_needed = max(target_length - len(tokens), 0)
        pad_token = self._vocab[self._pad_char]
        tokens = tokens + [pad_token] * padding_needed
        mask = mask + [0] * padding_needed


        return tokens, mask
    
    def decode(
        self,
        token_ids: list[int],
        **kwargs
    ) -> str:

        text = "".join([self._reverse_vocab[i] for i in token_ids])
        text = text.replace(self._pad_char, "")
        return text
    
    