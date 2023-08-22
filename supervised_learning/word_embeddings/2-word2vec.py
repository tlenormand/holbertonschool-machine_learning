#!/usr/bin/env python3
""" Bag Of Words """

import gensim
from gensim.models import Word2Vec


def word2vec_model(sentences, size=100, min_count=5, window=5, negative=5, cbow=True, iterations=5, seed=0, workers=1):
    """ Creates and trains a gensim word2vec model """
    model = Word2Vec(
        sentences=sentences,
        sg=cbow,
        negative=negative,
        window=window,
        min_count=min_count,
        workers=workers,
        seed=seed,
        size=size
    )

    model.train(
        sentences,
        epochs=iterations,
        total_examples=model.corpus_count
    )

    return model
