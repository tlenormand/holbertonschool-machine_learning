#!/usr/bin/env python3
""" Q-learning """

import numpy as np

epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """[Function that performs Q-learning]

    Args:
        env ([type]): [description]
        Q ([type]): [description]
        episodes (int, optional): [description]. Defaults to 5000.
        max_steps (int, optional): [description]. Defaults to 100.
        alpha (float, optional): [description]. Defaults to 0.1.
        gamma (float, optional): [description]. Defaults to 0.99.
        epsilon (int, optional): [description]. Defaults to 1.
        min_epsilon (float, optional): [description]. Defaults to 0.1.
        epsilon_decay (float, optional): [description]. Defaults to 0.05.

    Returns:
        [type]: [description]
    """
    total_rewards = []
    max_epsilon = epsilon

    for episode in range(episodes):
        state, info = env.reset()
        terminated = False
        rewards_current_episode = 0

        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            observation, reward, terminated, truncated, info = env.step(action)

            if terminated is True and reward == 0:
                reward = -1

            print("1:",Q[state, action], epsilon)
            Q[state, action] = Q[state, action] * (1 - alpha) + alpha * (reward + gamma * np.max(Q[observation, :]))
            print("2:",Q[state, action])
            state = observation
            rewards_current_episode += reward

            if terminated is True:
                break

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-epsilon_decay * episode)
        total_rewards.append(rewards_current_episode)

    return Q, total_rewards
