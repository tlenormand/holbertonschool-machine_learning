#!/usr/bin/env python3
""" Q-learning """

import numpy as np


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """ Q-learning """
    rewards = []
    max_epsilon = epsilon

    for episode in range(episodes):
        state = env.reset()
        done = False
        current_reward = 0

        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            new_state, reward, done, info = env.step(action)

            if done and reward == 0:
                reward = -1

            Q[state, action] = Q[state, action] * (1 - alpha) + alpha * (reward + gamma * np.max(Q[new_state, :]))

            state = new_state
            current_reward += reward

            if done:
                break

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay * episode)
        rewards.append(current_reward)

    return Q, rewards
