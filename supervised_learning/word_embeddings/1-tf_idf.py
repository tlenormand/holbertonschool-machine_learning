#!/usr/bin/env python3
""" Bag Of Words """

from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """ Creates a bag of words embedding matrix """
    if vocab is None:
        vectorizer = TfidfVectorizer()
    else:
        vectorizer = TfidfVectorizer(vocabulary=vocab)

    # Appliquer le vectorizer aux phrases pour obtenir la représentation TF-IDF
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler.fit_transform
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Obtenir les noms des fonctionnalités (mots)
    feature_names = vectorizer.get_feature_names_out()

    return tfidf_matrix.toarray(), feature_names
