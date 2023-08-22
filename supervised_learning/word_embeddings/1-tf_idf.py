#!/usr/bin/env python3
""" Bag Of Words """

from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """ Creates a bag of words embedding matrix """
    vec = TfidfVectorizer(use_idf=True, vocabulary=vocab)
    X = vec.fit_transform(sentences)

    embeddings = X.toarray()
    features = vec.get_feature_names()

    return embeddings, features
