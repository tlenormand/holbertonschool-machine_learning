#!/usr/bin/env python3
import os
import random
import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

WINDOW_LENGTH = 4
INPUT_SHAPE = (WINDOW_LENGTH, 210, 160)  # (WINDOW_LENGTH, height, width)
ATARI_INPUTS = {
    0: 'NOOP',
    1: 'FIRE',
    2: 'UP',
    3: 'RIGHT',
    4: 'LEFT',
    5: 'DOWN',
    6: 'UPRIGHT',
    7: 'UPLEFT',
    8: 'DOWNRIGHT',
    9: 'DOWNLEFT',
    10: 'UPFIRE',
    11: 'RIGHTFIRE',
    12: 'LEFTFIRE',
    13: 'DOWNFIRE',
    14: 'UPRIGHTFIRE',
    15: 'UPLEFTFIRE',
    16: 'DOWNRIGHTFIRE',
    17: 'DOWNLEFTFIRE'
}

BREAKOUT_INPUTS = {
    0: 0,
    1: 1,
    2: 3,
    3: 4,
}


class Model():
    def __init__(self, load_model=False):
        self.model_path = 'models/'
        self.extension = '.h5'
        self.model_name = 'breakout'
        self.hyperparameters = {
            'learning_rate': 0.0001,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_min': 0.01,
            'epsilon_decay_steps': 10000,
            'batch_size': 32,
            'target_update_frequency': 1000,
            'epochs': 10,
        }
        self.loss = tf.keras.losses.Huber()
        self.metrics = ['accuracy']

        self.target_model = None  # created with model in create_model()
        self.model = self.load_model() if load_model else self.create_model()

    def save_model(self, model, model_type):
        if not os.path.isdir(self.model_path):
            os.mkdir(self.model_path)
        self.model.save(self.model_path + model_type + self.model_name + self.extension)

    def load_model(self, model_type="target_model"):
        if not os.path.isdir(self.model_path):
            os.mkdir(self.model_path)
        self.target_model = models.load_model(self.model_path + model_type + self.model_name + self.extension)
        model = models.load_model(self.model_path + model_type + self.model_name + self.extension)

        print("Model loaded:", model.summary())

        return model

    def create_model(self):
        model = models.Sequential([
            layers.Permute((2, 3, 1), input_shape=INPUT_SHAPE),  # (WINDOW_LENGTH, height, width) => (height, width, WINDOW_LENGTH)
            layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu'),
            layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dense(4)
        ])

        self.target_model = models.Sequential([
            layers.Permute((2, 3, 1), input_shape=INPUT_SHAPE),  # (WINDOW_LENGTH, height, width) => (height, width, WINDOW_LENGTH)
            layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu'),
            layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dense(4)
        ])

        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.hyperparameters['learning_rate']),
            loss=self.loss,
            metrics=self.metrics
        )

        print("Model created:", model.summary())

        return model

    def predict(self, states):
        return self.model.predict(
            states,
            verbose=0,
        )

    def random_sample(self):
        return random.sample(self.memory, self.hyperparameters['batch_size'])


class Memory():
    def __init__(self):
        self.memory = []
        self.max_memory = 1000
        self.deleted_item_memory = 0

    def preprocess_state(self, state):
        # if already numpy array
        if isinstance(state, np.ndarray):
            return state

        return np.array(state)

    def add(self, state, action, reward, next_state, done, step):
        new_states = np.zeros(INPUT_SHAPE)
        new_states[0] = self.preprocess_state(state)

        self.memory.append(np.array([new_states, action, reward, next_state, done], dtype=object))

        for i in range(WINDOW_LENGTH - 1):
            if len(self.memory) > i + 1:
                self.memory[step - (i + 1) - self.deleted_item_memory][0][i + 1] = self.preprocess_state(state)

    def get_states(self, step):
        step -= 1  # step 0 is the first state
        print("STEP::", step)
        print("self.memory::", len(self.memory))

        if len(self.memory) <= 0:
            return None

        return self.memory[step - 1 - self.deleted_item_memory][0]


class DQNAgent():
    def __init__(self, epsilon_start=1):
        self.hyperparameters = {
            'learning_rate': 0.0001,
            'gamma': 0.99,
            'epsilon_start': epsilon_start,
            'epsilon_curent': epsilon_start,
            'epsilon_min': 0.01,
            'epsilon_decay_steps': 1,
            'epsilon_decay': 0.90,
            'batch_size': 32,
            'target_update_frequency': 1000,
        }
        self.Memory = Memory()

    def get_states(self, state, step):
        states = self.Memory.get_states(step)
        permuted_states = np.reshape(states, (1, states.shape[0], states.shape[1], states.shape[2]))
        return permuted_states

    def action_to_key(self, action):
        keys_list = list(BREAKOUT_INPUTS.keys())
        values_list = list(BREAKOUT_INPUTS.values())

        return keys_list[values_list.index(action)]

    def action_to_value(self, action):
        keys_list = list(BREAKOUT_INPUTS.keys())
        values_list = list(BREAKOUT_INPUTS.values())

        return values_list[action]

    def epsilon_greedy(self, Model, state, episode, step, action_space_n):
        if random.random() < self.hyperparameters['epsilon_curent']:
            # Exploration
            print('Exploration')
            action = np.random.choice(list(BREAKOUT_INPUTS.keys()))
        else:
            # Exploitation
            print('Exploitation')
            states = self.get_states(self.Memory, step)
            q_values = Model.predict(states)
            action = np.argmax(q_values)
            print("action: ", action)

        action = self.action_to_value(action)

        return action
    
    def build_next_state(self, state, next_state):
        new_state = np.zeros((1,) + INPUT_SHAPE)
        new_state[0][0] = next_state

        for i in range(WINDOW_LENGTH - 1):
            if len(self.Memory.memory) > i + 1:
                new_state[0][i + 1] = state[0][i]

        return new_state

    def experience_replay(self, Model):
        if len(self.Memory.memory) < Model.hyperparameters['batch_size']:
            return

        states = []
        targets = []

        len_memory = len(self.Memory.memory)
        while len_memory > Model.hyperparameters['batch_size']:
            minibatch = random.sample(self.Memory.memory, Model.hyperparameters['batch_size'])
            len_memory -= Model.hyperparameters['batch_size']

            for state, action, reward, next_state, done in minibatch:
                state = state / 255
                next_state = self.build_next_state(state, next_state)
                target = reward

                if not done:
                    best_future_action = np.argmax(Model.predict(next_state))
                    print(best_future_action)
                    target = reward + Model.hyperparameters['gamma'] * np.argmax(Model.target_model.predict(next_state, verbose=0)[0])  # [best_future_action]
                    print(reward)
                    print(Model.hyperparameters['gamma'])
                    print(Model.target_model.predict(next_state, verbose=0)[0][best_future_action])
                    print(np.argmax(Model.target_model.predict(next_state, verbose=0)[0]))
                    print(Model.target_model.predict(next_state, verbose=0)[0])
                    print("target::", target)

                state = np.expand_dims(state, axis=0)
                target_vector = Model.predict(state)[0]
                print("target_vector",target_vector)

                action = self.action_to_key(action)
                print("action",action)

                target_vector[action] = target
                print("target_vector[action]",target_vector[action])
                print("target_vector",target_vector)
                print()
                print()

                states.append(state[0])

                targets.append(target_vector)

        print('states: ', len(states), states[0].shape)
        Model.model.fit(np.array(states), np.array(targets), epochs=8, verbose=1)

        if len(states) > self.Memory.max_memory:
            excess = len(states) - self.Memory.max_memory
            while excess > 0:
                states.pop(0)
                targets.pop(0)
                excess -= 1

        if len(self.Memory.memory) > self.Memory.max_memory:
            excess = len(self.Memory.memory) - self.Memory.max_memory
            print("len(self.Memory.memory)::", len(self.Memory.memory))

            for _ in range(excess):
                self.Memory.memory.pop(0)
                self.Memory.deleted_item_memory += 1

        print("len(self.Memory.memory)::",len(self.Memory.memory))
        print("self.Memory.deleted_item_memory::", self.Memory.deleted_item_memory)

        # decay epsilon
        if self.hyperparameters['epsilon_curent'] > self.hyperparameters['epsilon_min']:
            self.hyperparameters['epsilon_curent'] *= self.hyperparameters['epsilon_decay']

        print('epsilon: ', self.hyperparameters['epsilon_curent'])

        # Model.train(np.array(states), np.array(targets))

class AtariGame():
    def __init__(self, render_mode='rgb_array'):
        self.game = 'Breakout-v5'
        self.path_game = 'ALE/'
        self.train_episodes = 1000
        self.play_episodes = 10
        self.total_reward = 0
        self.episode_reward = 0
        self.episode = 0
        self.step = 0


        self.env = self.make_env(render_mode)
        self.Model = Model()
        self.DQNAgent = DQNAgent()

        self.env.reset()

    def make_env(self, render_mode):
        return gym.make(
            self.path_game + self.game,
            obs_type='grayscale',
            frameskip=4,
            repeat_action_probability=0.25,
            full_action_space=True,
            render_mode=render_mode,  # human, rgb_array, single_rgb_array...
        )

    def train(self):
        for episode in range(self.train_episodes):
            state = self.env.reset()[0]
            self.episode_reward = 0
            self.episode = episode
            termination = False

            while not termination:
                action = DQNAgent.epsilon_greedy(self.DQNAgent, self.Model, state, self.episode, self.step, self.env.action_space.n)
                next_state, reward, termination, truncation, info = self.env.step(action)
                self.step += 1
                self.episode_reward += reward
                self.total_reward += reward

                print("self.episode_reward::", self.episode_reward)
                print("self.total_reward::", self.total_reward)

                states = self.DQNAgent.Memory.get_states(self.step)
                if states is None:
                    states = np.zeros((1,) + INPUT_SHAPE)
                    states[0][0] = state

                self.DQNAgent.Memory.add(state, action, reward, next_state, termination, self.step)

                state = next_state

            # fit the model
            self.DQNAgent.experience_replay(self.Model)

            # update target model
            self.Model.target_model.set_weights(self.Model.model.get_weights())

            # save model
            self.Model.save_model(self.Model.model, 'model')
            self.Model.save_model(self.Model.target_model, 'target_model')

        self.env.close()


    def play(self):
        for episode in range(self.play_episodes):
            state = self.env.reset()[0]
            self.episode_reward = 0
            self.episode = episode
            termination = False

            while not termination:
                print("len(self.DQNAgent.Memory.memory)::",len(self.DQNAgent.Memory.memory))
                states = self.DQNAgent.Memory.get_states(self.step)
                if states is None:
                    states = np.zeros((1,) + INPUT_SHAPE)
                    states[0][0] = state

                print("states::", np.array([states]).shape)

                if len(self.DQNAgent.Memory.memory) > 10:
                    action = np.argmax(self.Model.predict(np.array([states])))
                else:
                    action = np.random.choice(list(BREAKOUT_INPUTS.keys()))

                next_state, reward, termination, truncation, info = self.env.step(action)
                self.step += 1
                self.episode_reward += reward
                self.total_reward += reward

                self.DQNAgent.Memory.add(state, action, reward, next_state, termination, self.step)

                state = next_state

        self.env.close()


if __name__ == '__main__':
    AG = AtariGame(render_mode=None)
    AG.train()
    # AG.play()

    # AG.env.render()
