r"""
Pre-process the data.
"""

import torchtext
from sentarget.datasets import NoReCfine


class Process:
    r"""

    """

    def __init__(self, train_data, eval_data, test_data, fields=None):

        self.train_data, self.eval_data, self.test_data = train_data, eval_data, test_data
        self.fields = fields

    @classmethod
    def load(cls, fields=None):
        r"""Load the data.

        Args:
            fields:

        Returns:

        """
        text = torchtext.data.Field(lower=True, include_lengths=True, batch_first=True)
        label = torchtext.data.Field(batch_first=True)
        fields = fields if fields is not None else [("text", text), ("label", label)]
        train_data, eval_data, test_data = NoReCfine.splits(fields)
        return Process(train_data, eval_data, test_data, fields=fields)



