======
Set-up
======

Install
=======

Download this repository from **GitHub**:

.. code-block:: pycon

    $ git clone https://github.uio.no/arthurd/in5550-exam
    $ cd in5550-exam


The dataset is part of the repository, however you will need to give access to word embeddings.
You can either download the *Norwegian-Bokmaal CoNLL17* a.k.a the ``58.zip`` file from the nlpl_ website,
or provide them from saga_ server.

Make sure that you decode this file with ``encoding='latin1``.


.. _nlpl: http://vectors.nlpl.eu/repository/#

.. _saga: https://documentation.sigma2.no/


Baseline
========

To run the ``baseline.py`` script, use ``device=torch.device('cpu')`` as the ``cuda`` version was not implemented.

.. code-block:: pycon

    $ python baseline.py


Additional arguments can be provided, as:

* ``--NUM_LAYERS``: number of hidden layers for BiLSTM
* ``--HIDDEN_DIM``: dimensionality of LSTM layers
* ``--BATCH_SIZE``: number of examples to include in a batch
* ``--DROPOUT``: dropout to be applied after embedding layer
* ``--EMBEDDING_DIM``: dimensionality of embeddings
* ``--EMBEDDINGS``: location of pretrained embeddings
* ``--TRAIN_EMBEDDINGS``: whether to train or leave fixed
* ``--LEARNING_RATE``: learning rate for ADAM optimizer
* ``--EPOCHS``: number of epochs to train model