#!/usr/bin/env python3
""" Play """

import numpy as np


def play(env, Q, max_steps=100):
    """[Function that has the trained agent play an episode]

    Args:
        env ([type]): [description]
        Q ([type]): [description]
        max_steps (int, optional): [description]. Defaults to 100.

    Returns:
        [type]: [description]
    """
    state, info = env.reset()

    for step in range(max_steps):
        print("coucou")
        action = np.argmax(Q[state, :])
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated is True:
            break

        state = observation

        # display state in console
        print(state)

    return reward
