import os
import sys
import sentencepiece


class Tokenizer:
    def __init__(self, path):
        self.sp = sentencepiece.SentencePieceProcessor()
        self.sp.Load(path)

        self.vocab_size = self.sp.vocab_size()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.pad_id = self.sp.pad_id()

    def encode(self, text, prepend_bos=False, append_eos=False):
        return (
            ([self.bos_id] if prepend_bos else [])
            + self.sp.EncodeAsIds(text)
            + ([self.eos_id] if append_eos else [])
        )
