
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
    ) -> tuple[list[int], list[int], list[int]]:
    
        token_ids = [self._vocab[c] for c in text]
        input_token_ids = token_ids[:-1]
        target_token_ids = token_ids[1:]
        mask = [1] * len(input_token_ids)

        padding_needed = max(target_length - len(token_ids), 0)
        pad_token = self._vocab[self._pad_char]
        input_token_ids = input_token_ids + [pad_token] * padding_needed
        target_token_ids = target_token_ids + [pad_token] * padding_needed
        mask = mask + [0] * padding_needed


        return input_token_ids, target_token_ids, mask
    
    def decode(
        self,
        token_ids: list[int],
        **kwargs
    ) -> str:

        text = "".join([self._reverse_vocab[i] for i in token_ids])
        text = text.replace(self._pad_char, "")
        return text
    
    