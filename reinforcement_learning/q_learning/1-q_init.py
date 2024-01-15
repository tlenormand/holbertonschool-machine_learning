#!/usr/bin/env python3

import numpy as np


def q_init(env):
    print("ici:",env.observation_space.n, env.action_space.n)
    return np.zeros((env.observation_space.n, env.action_space.n))
