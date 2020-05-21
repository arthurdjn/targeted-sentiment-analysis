from collections import defaultdict
import os

import torch
from torch.nn.utils.rnn import pack_sequence
import torchtext


class Vocab(defaultdict):
    """
    This function creates the vocabulary dynamically. As you call ws2ids, it updates the vocabulary with any new tokens.
    """

    def __init__(self, train=True):
        super().__init__(lambda: len(self))
        self.train = train
        self.PAD = "<PAD>"
        self.UNK = "<UNK>"
        # set UNK token to 1 index
        self[self.PAD]
        self[self.UNK]

    def ws2ids(self, ws):
        """ If train, you can use the default dict to add tokens
            to the vocabulary, given these will be updated during
            training. Otherwise, we replace them with UNK.
        """
        if self.train:
            return [self[w] for w in ws]
        else:
            return [self[w] if w in self else 1 for w in ws]

    def ids2sent(self, ids):
        idx2w = dict([(i, w) for w, i in self.items()])
        return [idx2w[int(i)] if int(i) in idx2w else "<UNK>" for i in ids]


class Split(object):
    r"""
    Split a ``ConllDataset`` datasets.
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def pack_words(self, ws):
        return pack_sequence(ws)

    def collate_fn(self, batch):
        batch = sorted(batch, key=lambda item: len(item[0]), reverse=True)

        raws = [raw for raw, word, target, idx in batch]
        words = pack_sequence([word for raw, word, target, idx in batch])
        targets = pack_sequence([target for raw, word, target, idx in batch])
        idxs = [idx for raw, word, target, idx in batch]

        return raws, words, targets, idxs


class ConllDataset(object):
    r"""
    Read a .conll datasets.
    """
    def __init__(self, vocab):
        self.vocab = vocab
        self.label2idx = {"O": 0, "B-targ-Positive": 1, "I-targ-Positive": 2,
                          "B-targ-Negative": 3, "I-targ-Negative": 4}

    def get_split(self, data_file):
        text = torchtext.data.Field(lower=False, include_lengths=True, batch_first=True)
        label = torchtext.data.Field(batch_first=True)
        data = torchtext.datasets.SequenceTaggingDataset(data_file,
                                                         fields=[("text", text),
                                                                 ("label", label)])
        data_split = [(item.text,
                       torch.LongTensor(self.vocab.ws2ids(item.text)),
                       torch.LongTensor([self.label2idx[l] for l in item.label]),
                       idx) for idx, item in enumerate(data)]
        return Split(data_split)
