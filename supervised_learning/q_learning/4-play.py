#!/usr/bin/env python3
""" Q-learning """

import numpy as np


def play(env, Q, max_steps=100):
    """ Q-learning """
    state = env.reset()
    env.render()

    for step in range(max_steps):
        action = np.argmax(Q[state, :])
        new_state, reward, done, info = env.step(action)
        env.render()

        if done:
            break

        state = new_state

    return reward
