import pandas as pd
import torch


class Tokenizer:
    def __init__(
        self,
        use_gpt: bool,
        file_path="./data/dataset5/step2/dict_id2word.pkl",
    ):
        """Initializes the tokenizer."""
        self.dict_id2word = pd.read_pickle(file_path)
        self.use_gpt = use_gpt
        self.vocab_size = len(self.dict_id2word)
        self.start_token_id = self.vocab_size
        self.pad_token_id = self.vocab_size if use_gpt else self.vocab_size + 1
        self.max_label_length = 151

    def encode(self, tokens: list):
        encoded_tokens = torch.tensor(tokens, dtype=torch.long)
        len_label = len(encoded_tokens)
        dif = self.max_label_length - len_label
<<<<<<< HEAD
        encoded_tokens = torch.cat((encoded_tokens, torch.full(
            (dif,), self.pad_token_id, dtype=torch.long)))
        if self.use_gpt:
            encoded_tokens = torch.cat(
                [torch.tensor([self.start_token_id],
                              dtype=torch.long), encoded_tokens]
=======
        encoded_tokens = torch.cat(
            (encoded_tokens, torch.full((dif,), self.pad_token_id, dtype=torch.long))
        )
        if self.use_gpt:
            encoded_tokens = torch.cat(
                [torch.tensor([self.start_token_id], dtype=torch.long), encoded_tokens]
>>>>>>> 04fac6932c8843d0673604ce65b4966ec82d6d17
            )
        return encoded_tokens

    def decode(self, token_ids):
        """
        Decodes a list of token IDs back to a string.

        :param token_ids: List of integers representing token IDs.
        :return: Decoded string.
        """
        return "".join(self.decode_seq(token_ids))

    def decode_seq(self, token_ids):
        return [self.dict_id2word[id.item()] for id in token_ids if id.item() in self.dict_id2word]

    def get_vocab_size(self):
        return self.vocab_size + 1 if self.use_gpt else self.vocab_size + 2

    def get_max_label_length(self):
        return self.max_label_length
