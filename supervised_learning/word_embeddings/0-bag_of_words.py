#!/usr/bin/env python3
""" Bag Of Words """


def bag_of_words(sentences, vocab=None):
    """ creates a bag of words embedding matrix """
    if vocab is None:
        vocab = []
        for sentence in sentences:
            for word in sentence.split():
                if word not in vocab:
                    vocab.append(word)
        vocab.sort()

    array = []

    for sentence in sentences:
        temp = []
        for word in vocab:
            if word in sentence:
                temp.append(1)
            else:
                temp.append(0)
        array.append(temp)

    return array, vocab
