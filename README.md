[![docs](https://img.shields.io/badge/github%20page-up-green)](https://arthurdjn.github.io/targeted-sentiment-analysis/)

# Status

### Documentation

The official docs can be foud [here](https://pages.github.uio.no/arthurd/in5550-exam).
We used _PyTorch_ on this project, even for the documentation builds.

### Code

| Development                  | Status      | Feature                                                                                  |
| ---------------------------- | ----------- | ---------------------------------------------------------------------------------------- |
| Baseline                     | finished    | <ul><li>[x] DataLoader</li><li>[x] Word2Vec</li><li>[x] BiLSTM</li></ul>                 |
| Alternative Label Encoding   | not started | <ul><li>[ ] BIOUL</li></ul>                                                              |
| Pipeline vs Joint prediction | not started | <ul><li>[ ] Pipeline</li><li>[ ] Joint Prediction</li><li>[ ] Comparison</li></ul>       |
| Architecture Impact          | in progress | <ul><li>[x] LSTM</li><li>[x] GRU</li><li>[ ] Character Level</li><li>[x] Depth</li></ul> |
| Pretrained Embeddings        | in progress | <ul><li>[ ] ELMo</li><li>[x] BERT</li><li>[x] Multilingual BERT</li></ul>                |
| Error Analysis               | finished    | <ul><li>[x] Confusion Matrix</li><li>[x] Common Errors</li></ul>                         |

# Norwegian Data

For this targeted sentiment analysis, we used a training dataset in Norwegian with corresponding word embeddings.

### NoRec Dataset

We will be working with the recently released [NoReCfine](https://www.researchgate.net/publication/337671672_A_Fine-grained_Sentiment_Dataset_for_Norwegian),
a dataset for finegrained sentiment analysis in Norwegian.
The texts in the dataset have been annotated with respect to polar expressions, targets and holders of opinion but
we will here be focusing on identification of targets and their polarity only.
The underlying texts are taken from a corpus of professionally authored reviews
from multiple news-sources and across a wide variety of domains, including
literature, games, music, products, movies and more.

### NLPL Word Embeddings

the word embeddings used are taken from the [NLPL](http://vectors.nlpl.eu/repository/#) datasets,
using the _Norwegian-Bokmaal CoNLL17_ corpus, with a vocabulary size of 1,182,371.

# Getting Started

### Set-up

Download this repository:

```
$ git clone https://github.uio.no/arthurd/wnnlp
```

The dataset is part of the repository, however you will need to give access to word embeddings.
You can either download the _Norwegian-Bokmaal CoNLL17_ a.k.a the `58.zip` file from the [NLPL](http://vectors.nlpl.eu/repository/#) website,
or provide them from [SAGA](https://documentation.sigma2.no/) server.

Make sure that you decode this file with `encoding='latin1`.

### Baseline

```
$ python baseline.py --NUM_LAYERS         number of hidden layers for BiLSTM
                     --HIDDEN_DIM         dimensionality of LSTM layers
                     --BATCH_SIZE         number of examples to include in a batch
                     --DROPOUT            dropout to be applied after embedding layer
                     --EMBEDDING_DIM      dimensionality of embeddings
                     --EMBEDDINGS         location of pretrained embeddings
                     --TRAIN_EMBEDDINGS   whether to train or leave fixed
                     --LEARNING_RATE      learning rate for ADAM optimizer
                     --EPOCHS             number of epochs to train model
```

### Grid Search

The grid search is currently availabel for the `BiLSTM` and `BiGRU` models.
You can access through their inner parameters (and hyper parameters as well) through the `gridsearch.ini`
configuration file. This file is divided into multiple sections, corresponding to diverse parameters, and
you will find more information there.

To run the gridsearch algorithm, simply modify the above parameters and run:

```
$ python gridsearch.py --conf   PATH_TO_CONFIGURATION_FILE
```

### Evaluation

To test and evaluate a saved model, use the `eval.py` script as follow:

```
$ python eval.py --model  PATH_TO_SAVED_MODEL
                 --data   PATH_TO_EVAL_DATA
```
