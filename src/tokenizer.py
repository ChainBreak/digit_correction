
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

        self.pad_token = self._vocab["<PAD>"]
        self.sos_token = self._vocab["<SOS>"]
        self.eos_token = self._vocab["<EOS>"]
       
    def __len__(self):
        return len(self._vocab)
        
    def encode(
        self,
        text: str,
    ) -> list[int]:
    
        token_ids = [self._vocab[c] for c in text]
        
        token_ids = [self.sos_token] + token_ids + [self.eos_token]

        return token_ids

    
    def decode(
        self,
        token_ids: list[int],
    ) -> str:

        # Find the first index where the token id is equal to self._sos_token
        
        sos_index = token_ids.index(self.sos_token)
        try:
            eos_index = token_ids.index(self.eos_token)
        except ValueError:
            eos_index = len(token_ids)

        token_ids = token_ids[sos_index + 1:eos_index]

        text = "".join([self.id_to_char(i) for i in token_ids])
        
        return text

    def id_to_char(self, index: int) -> str:
        if index < 0 or index >= len(self._reverse_vocab):
            return "?"
        return self._reverse_vocab[index]
    
    