r"""
Run a simple grid search algorithm.
"""

import configparser
from argparse import ArgumentParser

import numpy
import torch
from torchtext import data
from torchtext.vocab import Vectors

import sentarget
from sentarget.datasets import NoReCfine
from sentarget.tuner import Tuner


def gridsearch(options={}, params_hyper={}, params_model={}, params_optim={}, params_loss={}):
    """Run the grid search algorithms on the CONLL dataset provided.

    Args:
        options (dict): general options.
        params_hyper (dict): hyper parameters to tune.
        params_model (dict): model's parameters to tune.
        params_optim (dict): optimizer parameters to tune.
        params_loss (dict): criterion parameters to tune.

    """

    # 1/ Load the data
    TEXT = data.Field(lower=False, include_lengths=True, batch_first=True)
    LABEL = data.Field(batch_first=True, unk_token=None)
    FIELDS = [("text", TEXT), ("label", LABEL)]
    train_data, eval_data, test_data = NoReCfine.splits(FIELDS)

    # 2/ Build the vocab
    VOCAB_SIZE = 1_200_000
    VECTORS_NAME = params_hyper['vectors_name']
    VECTORS_URL = params_hyper['vectors_url']
    VECTORS = Vectors(name=VECTORS_NAME, url=VECTORS_URL)
    TEXT.build_vocab(train_data, test_data, eval_data,
                     max_size=VOCAB_SIZE,
                     vectors=VECTORS,
                     unk_init=torch.Tensor.normal_)
    LABEL.build_vocab(train_data)

    # 3/ Load iterators
    BATCH_SIZE = params_hyper['batch_size']
    device = torch.device('cpu')
    train_iterator, eval_iterator, test_iterator = data.BucketIterator.splits((train_data, eval_data, test_data),
                                                                              batch_size=BATCH_SIZE,
                                                                              sort_within_batch=True,
                                                                              device=device)

    # Initialize the embedding layer
    if params_hyper['use_pretrained_embeddings']:
        params_model['embeddings'] = TEXT.vocab.vectors

    # 4/ Grid Search
    tuner = Tuner(params_hyper=params_hyper,
                  params_model=params_model,
                  params_loss=params_loss,
                  params_optim=params_optim,
                  options=options)

    # Search
    tuner.fit(train_iterator, eval_iterator)
    tuner.save(dirsaves=options['dirsaves'])


if __name__ == "__main__":
    # As there are a lot of customizable parameters (the grid search run on all module's parameters)
    # It is more readable to separate the configuration from the code.
    # The configuration file is a .ini format,
    # but you can create your own custom functions depending on the grid search algorithm that you need.

    parser = ArgumentParser()
    parser.add_argument('-c', '--conf', help="Path to the config.ini file to use.", action='store',
                        type=str, default='gridsearch.ini')
    args = parser.parse_args()

    # Read the configuration file
    config = configparser.ConfigParser()
    config.read(args.conf)

    options = {key: eval(value) for (key, value) in dict(config.items('Options')).items()}
    params_hyper = {key: eval(value) for (key, value) in dict(config.items('Hyper')).items()}
    params_model = {key: eval(value) for (key, value) in dict(config.items('Model')).items()}
    params_loss = {key: eval(value) for (key, value) in dict(config.items('Criterion')).items()}
    params_optim = {key: eval(value) for (key, value) in dict(config.items('Optimizer')).items()}

    # Run the gridsearch
    gridsearch(
        params_hyper=params_hyper,
        params_model=params_model,
        params_loss=params_loss,
        params_optim=params_optim,
        options=options
    )
