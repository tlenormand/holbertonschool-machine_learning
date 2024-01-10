#!/usr/bin/env python3
""" Question Answering """

import tensorflow as tf
import tensorflow_hub as hub
import transformers as ts


def question_answer(question, reference):
    """ Method that finds a snippet of text within a reference document
        to answer a question.

    Args:
        question: (str) containing the question to answer.
        reference: (str) containing the reference document
            from which to find the answer.

    Returns:
        answer: (str) containing the answer.
    """
    tokenizer = ts.BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    # tokenize question and reference
    question_token = tokenizer.tokenize(question)
    paragraph_token = tokenizer.tokenize(reference)

    # add special tokens
    tokens = ['[CLS]'] + question_token + ['[SEP]'] + paragraph_token + ['[SEP]']
    # convert tokens to ids
    input_word_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * len(input_word_ids)
    input_type_ids = [0] * (1 + len(question_token) + 1) + [1] * (len(paragraph_token) + 1)

    input_word_ids, input_mask, input_type_ids = map(lambda t: tf.expand_dims(
        tf.convert_to_tensor(t, dtype=tf.int32), 0),
        (input_word_ids, input_mask, input_type_ids)
    )
    outputs = model([input_word_ids, input_mask, input_type_ids])

    # using `[1:]` will enforce an answer. `outputs[0][0][0]` is the ignored '[CLS]' token logit
    short_start = tf.argmax(outputs[0][0][1:]) + 1
    short_end = tf.argmax(outputs[1][0][1:]) + 1

    answer_tokens = tokens[short_start: short_end + 1]
    answer = tokenizer.convert_tokens_to_string(answer_tokens)

    return answer if len(answer) > 1 else None
