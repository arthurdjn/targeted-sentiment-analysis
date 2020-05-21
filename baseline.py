"""
Run the baseline script.
"""

import torch
from torch.utils.data import DataLoader
import numpy as np

from baseline.dataset import Vocab, ConllDataset
from baseline.word2vec import Word2Vec
from baseline.model import BiLSTM

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--NUM_LAYERS", "-nl", default=1, type=int)
    parser.add_argument("--HIDDEN_DIM", "-hd", default=100, type=int)
    parser.add_argument("--BATCH_SIZE", "-bs", default=50, type=int)
    parser.add_argument("--DROPOUT", "-dr", default=0.01, type=int)
    parser.add_argument("--EMBEDDING_DIM", "-ed", default=100, type=int)
    parser.add_argument("--EMBEDDINGS", "-emb", default="word2vec/models.txt")
    parser.add_argument("--TRAIN_EMBEDDINGS", "-te", action="store_true")
    parser.add_argument("--LEARNING_RATE", "-lr", default=0.01, type=int)
    parser.add_argument("--EPOCHS", "-e", default=50, type=int)

    args = parser.parse_args()
    print(args)

    # Get embeddings (CHANGE TO GLOVE OR FASTTEXT EMBEDDINGS)
    embeddings = Word2Vec(args.EMBEDDINGS)
    w2idx = embeddings._w2idx

    # Create shared vocabulary for tasks
    vocab = Vocab(train=True)

    # Update with word2idx from pretrained embeddings so we don't lose them
    # making sure to change them by two to avoid overwriting the PAD and UNK
    # tokens at index 0 and 1
    with_unk = {}
    for word, idx in embeddings._w2idx.items():
        with_unk[word] = idx + 2
    vocab.update(with_unk)

    # Import datasets
    # This will update vocab with words not found in embeddings
    dataset = ConllDataset(vocab)

    train_iter = dataset.get_split("data/train.conll")
    dev_iter = dataset.get_split("data/dev.conll")
    test_iter = dataset.get_split("data/test.conll")

    # Create a new embedding matrix which includes the pretrained embeddings
    # as well as new embeddings for PAD UNK and tokens not found in the
    # pretrained embeddings.
    diff = len(vocab) - embeddings.vocab_length - 2
    PAD_UNK_embeddings = np.zeros((2, args.EMBEDDING_DIM))
    new_embeddings = np.zeros((diff, args.EMBEDDING_DIM))
    new_matrix = np.concatenate((PAD_UNK_embeddings,
                                 embeddings._matrix,
                                 new_embeddings))

    # Set up the data iterators for the LSTM models. The batch size for the dev
    # and test loader is set to 1 for the predict() and evaluate() methods
    train_loader = DataLoader(train_iter,
                              batch_size=args.BATCH_SIZE,
                              collate_fn=train_iter.collate_fn,
                              shuffle=True)

    dev_loader = DataLoader(dev_iter,
                            batch_size=1,
                            collate_fn=dev_iter.collate_fn,
                            shuffle=False)

    test_loader = DataLoader(test_iter,
                             batch_size=1,
                             collate_fn=test_iter.collate_fn,
                             shuffle=False)

    # Automatically determine whether to run on CPU or GPU
    device = torch.device('cpu')

    model = BiLSTM(word2idx=vocab,
                   embedding_matrix=new_matrix,
                   embedding_dim=args.EMBEDDING_DIM,
                   hidden_dim=args.HIDDEN_DIM,
                   device=device,
                   output_dim=5,
                   num_layers=args.NUM_LAYERS,
                   word_dropout=args.DROPOUT,
                   learning_rate=args.LEARNING_RATE,
                   train_embeddings=args.TRAIN_EMBEDDINGS)

    model.fit(train_loader, dev_loader, epochs=args.EPOCHS)

    binary_f1, propor_f1 = model.evaluate(test_loader)

    # For printing the predictions, we would prefer to see the actual labels,
    # rather than the indices, so we create and index to label dictionary
    # which the print_predictions method takes as input.

    idx2label = {i: l for l, i in dataset.label2idx.items()}
    model.print_predictions(test_loader,
                            outfile="predictions.conll",
                            idx2label=idx2label)
