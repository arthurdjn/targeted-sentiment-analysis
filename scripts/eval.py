"""
Main script used to run and test a model, for Targeted Sentment Ananalysis.
The dataset used should be taken from the lattest NoReCfine repository.
"""

import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchtext
from torchtext.datasets import SequenceTaggingDataset
from torchtext.vocab import Vectors

import numpy as np

import sentarget
from sentarget.datasets import NoReCfine
from sentarget.metrics import ConfusionMatrix
from sentarget.utils import describe_dict


class Eval:
    """
    Evaluate and test our model trained on the NoReCfine dataset.
    This class load and preprocess the data, and then evaluate the model.

    """

    def __init__(self, model_path='model.pt', data_path='data', device='cpu'):
        self.model_path = model_path
        self.data_path = data_path
        self.device = device

    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", "-m", default='model.pt', type=str,
                            help='Path to the saved pytorch model.')
        parser.add_argument("--data", "-d", default='data/test.conll', type=str,
                            help='Path to the dataset, in the same format as NoReC dataset.')
        args = parser.parse_args()
        return Eval(model_path=args.model, data_path=args.data)

    def run(self):
        """Preprocess and eval the model.

        """
        # Extract Fields from a CONLL dataset file
        TEXT = torchtext.data.Field(lower=False, include_lengths=True, batch_first=True)
        LABEL = torchtext.data.Field(batch_first=True, unk_token=None)
        FIELDS = [("text", TEXT), ("label", LABEL)]
        train_data, eval_data, test_data = NoReCfine.splits(FIELDS)
        data = SequenceTaggingDataset(self.data_path, FIELDS, encoding="utf-8", separator="\t")

        # Build the vocabulary
        VOCAB_SIZE = 1_200_000
        VECTORS = Vectors(name='model.txt', url='http://vectors.nlpl.eu/repository/20/58.zip')
        # Create the vocabulary for words embeddings
        TEXT.build_vocab(train_data,
                         max_size=VOCAB_SIZE,
                         vectors=VECTORS,
                         unk_init=torch.Tensor.normal_)
        LABEL.build_vocab(train_data)

        # General information
        text_length = [len(sentence) for sentence in list(data.text)]
        print(f"\nNumber of sentences in {self.data_path}: {len(text_length):,}")
        print(f'Number of words in {self.data_path}: {sum(text_length):,}')

        # Generate iterator made of 1 example
        BATCH_SIZE = 1
        device = torch.device(self.device)
        iterator = torchtext.data.BucketIterator(data,
                                                 batch_size=BATCH_SIZE,
                                                 sort_within_batch=True,
                                                 device=device)

        # Loss function
        criterion = nn.CrossEntropyLoss(ignore_index=0, weight=torch.tensor(
            [1, 0.06771941, 0.97660534, 0.97719714, 0.98922782, 0.98925029]))

        # Load the model
        model = torch.load(self.model_path)
        # Make sure the dictionary containing performances / scores is empty before running the eval method
        # model.reset()
        performance = model.evaluate(iterator, criterion, verbose=True)
        print(describe_dict(performance, sep_key=' | ', sep_val=': ', pad=True))
        confusion = ConfusionMatrix(data=performance['confusion'])
        print("confusion matrix:")
        print(np.array2string(confusion.normalize(), separator=',  ', precision=3, floatmode='fixed'))


if __name__ == "__main__":
    Eval.from_args().run()
