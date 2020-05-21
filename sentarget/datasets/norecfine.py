"""
The ``NoReCfine`` class defines the latest datasets used for targeted sentiment analysis.

.. code-block:: python

    # First, download the training / dev / test data
    train_data, dev_data, test_data = NoReCfine.splits(train_data="path_to_train",
                                                       dev_data="path_to_eval",
                                                       test_data="path_to_test")

"""

from torchtext.datasets import SequenceTaggingDataset


class NoReCfine(SequenceTaggingDataset):
    r"""This class defines the ``NoReCfine`` datasets,
    used on the paper *A Fine-grained Sentiment Dataset for Norwegian.*

    """

    @classmethod
    def splits(cls, fields, train_data="data/train.conll", dev_data="data/dev.conll", test_data="data/test.conll"):
        return NoReCfine(train_data, fields), NoReCfine(dev_data, fields), NoReCfine(test_data, fields)
