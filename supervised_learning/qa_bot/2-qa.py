#!/usr/bin/env python3
""" Create the loop """

question_answer = __import__('0-qa').question_answer


def load_reference():
    """ Method that loads the reference text

    Returns:
        reference: (str) containing the reference document
            from which to find the answer.
    """
    references_path = [
        'ZendeskArticles/PeerLearningDays.md'
    ]

    reference = ''
    for path in references_path:
        with open(path) as f:
            reference += f.read()

    return reference


def question_handler(question):
    """ Method that answers questions from a reference text

    Args:
        question: (str) containing the question to answer.

    Returns:
        -1: if the user exits.
        0: if the answer cannot be found.
        1: if the answer was found.
    """
    exit = ['exit', 'quit', 'goodbye', 'bye']

    if question in exit:
        print('A: Goodbye')
        return -1
    else:
        answer = question_answer(question, reference)

        if answer is None:
            print('A: Sorry, I do not understand your question.')
            return 0
        else:
            print(f'A: {answer}')

    return 1


def loop_QA():
    """ Create the prompt loop """
    while True:
        question = input('Q: ')
        response_status = question_handler(question.lower())

        # exit
        if response_status == -1:
            break


if __name__ == '__main__':
    reference = load_reference()
    loop_QA()
