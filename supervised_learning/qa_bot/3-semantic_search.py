#!/usr/bin/env python3
""" Semantic Search """

import os
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_text as text


def semantic_search(corpus_path, sentence):
    """ Method that performs semantic search on a corpus of documents

    Args:
        corpus_path: (str) the path to the corpus of reference documents on
            which to perform semantic search.
        sentence: (str) the sentence from which to perform semantic search.

    Returns:
        reference: (str) the reference text of the document most similar to
            sentence.
    """
    documents = [sentence]
    for filename in os.listdir(corpus_path):
        if filename.endswith('.md'):
            with open(corpus_path + '/' + filename, 'r', encoding='utf-8') as f:
                documents.append(f.read())

    model = hub.load('https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3')
    embeddings = model(documents)

    correlation = np.inner(embeddings, embeddings)
    closest = np.argmax(correlation[0, 1:])
    reference = documents[closest + 1]

    return reference
