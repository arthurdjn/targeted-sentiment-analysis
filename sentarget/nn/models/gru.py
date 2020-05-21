r"""
The Bilinear Recurrent network is a vanilla model used for targeted sentiment analysis,
and compared to more elaborated models.

Example:

.. code-block:: python

    # Defines the shape of the models
    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 128
    OUTPUT_DIM = len(LABEL.vocab)
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.25
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    model = BiGRU(INPUT_DIM,
                   EMBEDDING_DIM,
                   HIDDEN_DIM,
                   OUTPUT_DIM,
                   N_LAYERS,
                   BIDIRECTIONAL,
                   DROPOUT,
                   PAD_IDX)

"""

import time
import torch
import torch.nn as nn

from sentarget.metrics import ConfusionMatrix
from sentarget.utils import progress_bar
from .model import Model


class BiGRU(Model):
    r"""This bilinear model uses the `sklearn` template, i.e. with a fit method within the module.

    Make sure to add a criterion and optimizer when loading a model.

    * :attr:`input_dim` (int): input dimension, i.e. dimension of the incoming words.

    * :attr:`embedding_dim` (int): dimension of the word embeddigns.

    * :attr:`hidden_dim` (int): dimmension used to map words with the recurrent unit.

    * :attr:`output_dim` (int): dimension used for classification. This one should be equals to the number of classes.

    * :attr:`n_layers` (int): number of recurrent layers.

    * :attr:`bidirectional` (bool): if `True`, set two recurrent layers in the opposite direction.

    * :attr:`dropout` (float): ratio of connections set to zeros.

    * :attr:`pad_idx_text` (int): index of the `<pad>` text token.

    * :attr:`pad_idx_label` (int): index of the `<pad>` label token.

    * :attr:`embeddings` (torch.Tensor): pretrained embeddings, of shape ``(input_dim, embeddings_dim)``.


    Examples::

        >>> INPUT_DIM = len(TEXT.vocab)
        >>> EMBEDDING_DIM = 100
        >>> HIDDEN_DIM = 128
        >>> OUTPUT_DIM = len(LABEL.vocab)
        >>> N_LAYERS = 2
        >>> BIDIRECTIONAL = True
        >>> DROPOUT = 0.25
        >>> PAD_IDX_TEXT = TEXT.vocab.stoi[TEXT.pad_token]
        >>> PAD_IDX_LABEL = LABEL.vocab.stoi[LABEL.unk_token]

        >>> model = BiGRU(INPUT_DIM,
        ...                EMBEDDING_DIM,
        ...                HIDDEN_DIM,
        ...                OUTPUT_DIM,
        ...                N_LAYERS,
        ...                BIDIRECTIONAL,
        ...                DROPOUT,
        ...                pad_idx_text=PAD_IDX_TEXT,
        ...                pad_idx_label=PAD_IDX_LABEL)

        >>> criterion = nn.CrossEntropyLoss()
        >>> optimizer = metrics.Adam(model.parameters())
        >>> model.fit(50, train_data, eval_data, criterion, optimizer)

    """

    def __init__(self,
                 input_dim,
                 embedding_dim=100,
                 hidden_dim=128,
                 output_dim=7,
                 n_layers=2,
                 bidirectional=True,
                 dropout=0.25,
                 pad_idx_text=1,
                 unk_idx_text=0,
                 pad_idx_label=0,
                 embeddings=None):
        super().__init__()
        # dimensions
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim

        # modules
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=pad_idx_text)
        self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers, bidirectional=bidirectional, batch_first=True,
                            dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

        if embeddings is not None:
            ignore_index = [idx for idx in [pad_idx_text, unk_idx_text] if idx is not None]
            self.init_embeddings(embeddings, ignore_index=ignore_index)

        # tokens
        self.pad_idx_text = pad_idx_text
        self.pad_idx_label = pad_idx_label
        self.unk_idx_text = unk_idx_text

    def init_embeddings(self, embeddings, ignore_index=None):
        r"""Initialize the embeddings vectors from pre-trained embeddings vectors.

        .. Warning::

            By default, the embeddings will set to zero the tokens at indices 0 and 1,
            that should corresponds to <pad> and <unk>.


        Examples::

            >>> # TEXT: field used to extract text, sentences etc.
            >>> PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
            >>> UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
            >>> pretrained_embeddings = TEXT.vocab.vectors

            >>> model.init_embeddings(pretrained_embeddings, ignore_index=[PAD_IDX, UNK_IDX])


        Args:
            embeddings (torch.tensor): pre-trained word embeddings, of shape ``(input_dim, embedding_dim)``.
            ignore_index (int or iterable): if not `None`, set to zeros tensors at the indices provided.

        """
        self.embedding.weight.data.copy_(embeddings)
        if ignore_index is not None:
            if isinstance(ignore_index, int):
                self.embedding.weight.data[ignore_index] = torch.zeros(self.embedding_dim)
            elif isinstance(ignore_index, list) or isinstance(ignore_index, tuple):
                for index in ignore_index:
                    self.embedding.weight.data[index] = torch.zeros(self.embedding_dim)
            elif isinstance(ignore_index, dict):
                raise KeyError("Ambiguous `ignore_index` provided. "
                               "Please provide an iterable like a `list` or `tuple`.")

    def forward(self, text, length):
        r"""One forward step.

        .. note::

            The forward propagation requires text's length, so a padded pack can be applied to batches.

        Args:
            text (torch.tensor): text composed of word embeddings vectors from one batch.
            length (torch.tensor): vector indexing the lengths of `text`.


        Examples::

            >>> for batch in data_iterator:
            >>>     text, length = batch.text
            >>>     model.forward(text, length)

        """
        # Word embeddings
        embeddings = self.embedding(text)
        # Apply a dropout
        embedded = self.dropout(embeddings)
        # Pack and pad a batch
        packedembeds = nn.utils.rnn.pack_padded_sequence(embedded, length, batch_first=True)
        # Apply the recurrent cell
        packed_output, h_n = self.gru(packedembeds)
        # Predict
        output = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)[0]
        # Apply another dropout and a linear layer for classification tasks
        predictions = self.fc(self.dropout(output))

        return predictions

    def get_accuracy(self, y_tilde, y):
        r"""Computes the accuracy from a set of predictions and gold labels.

        .. note::

            The resulting accuracy does not count `<pad>` tokens.


        Args:
            y_tilde (torch.tensor): predictions.
            y (torch.tensor): gold labels.

        Returns:
            torch.tensor: the global accuracy, of shape 0.

        """
        non_pad_elements = (y != self.pad_idx_label).nonzero()
        correct = y_tilde[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
        accuracy = correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]])
        # Handles division by 0
        accuracy = accuracy if not torch.isnan(accuracy) else torch.tensor(0)
        return accuracy

    def run(self, iterator, criterion, optimizer, verbose=True):
        r"""Train one time the model on iterator data.

        Args:
            iterator (Iterator): iterator containing batch samples of data.
            criterion (Loss): loss function to measure scores.
            optimizer (Optimizer): optimizer used during training to update weights.
            verbose (bool): if `True` display a progress bar.

        Returns:
            dict: the performance and metrics of the training session.

        """
        # Initialize the variables
        start_time = time.time()
        epoch_loss = 0
        epoch_acc = 0
        class_labels = list(range(self.output_dim))
        class_labels.pop(self.pad_idx_label)
        confusion_matrix = ConfusionMatrix(labels=class_labels)

        # Train mode
        self.train()
        for (idx, batch) in enumerate(iterator):
            optimizer.zero_grad()
            # One forward step
            text, length = batch.text
            y_hat = self.forward(text, length)
            y_hat = y_hat.view(-1, y_hat.shape[-1])
            label = batch.label.view(-1)
            # Get the predicted classes
            y_tilde = y_hat.argmax(dim=1, keepdim=True)
            # Compute the loss and update the weights
            loss = criterion(y_hat, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # Default accuracy
            acc = self.get_accuracy(y_tilde, label)
            epoch_acc += acc.item()
            # Optional: display a progress bar
            if verbose:
                progress_bar(idx, len(iterator) - 1, prefix="Training:\t", start_time=start_time)

            # Update the confusion matrix
            confusion_matrix.update(label.long().numpy(), y_tilde.long().numpy())

        # Store the loss, accuracy and metrics in a dictionary
        results_train = {"loss": epoch_loss / len(iterator),
                         "accuracy": epoch_acc / len(iterator),
                         **confusion_matrix.to_dict()
                         }

        return results_train

    def evaluate(self, iterator, criterion, verbose=True):
        r"""Evaluate one time the model on iterator data.

        Args:
            iterator (Iterator): iterator containing batch samples of data.
            criterion (Loss): loss function to measure scores.
            verbose (bool): if `True` display a progress bar.

        Returns:
            dict: the performance and metrics of the training session.

        """
        # Initialize the variables
        start_time = time.time()
        epoch_loss = 0
        epoch_acc = 0
        class_labels = list(range(self.output_dim))
        class_labels.pop(self.pad_idx_label)
        confusion_matrix = ConfusionMatrix(labels=class_labels)

        # Eval mode
        self.eval()
        with torch.no_grad():
            for (idx, batch) in enumerate(iterator):
                # One forward step
                text, length = batch.text
                y_hat = self.forward(text, length)
                y_hat = y_hat.view(-1, y_hat.shape[-1])
                label = batch.label.view(-1)
                # Get the predicted classes
                y_tilde = y_hat.argmax(dim=1, keepdim=True)
                # Compute the loss
                loss = criterion(y_hat, label)
                epoch_loss += loss.item()
                # Default accuracy
                acc = self.get_accuracy(y_tilde, label)
                epoch_acc += acc.item()
                # Optional: display a progress bar
                if verbose:
                    progress_bar(idx, len(iterator) - 1, prefix="Evaluation:\t", start_time=start_time)

                # Update the confusion matrix
                confusion_matrix.update(label.long().numpy(), y_tilde.long().numpy())

        # Store the loss, accuracy and metrics in a dictionary
        results_eval = {"loss": epoch_loss / len(iterator),
                        "accuracy": epoch_acc / len(iterator),
                        **confusion_matrix.to_dict()
                        }

        return results_eval
