#!/usr/bin/env python3

import gym
import numpy as np
import tensorflow as tf

from classes.Logs import Logs
from classes.Memory import Memory
from classes.Model import Model
from global_variables import GAMES


class Atari():
    def __init__(self, game='Breakout-v5', load_model=False, epsilon=0.9, epsilon_min=0.1, render_mode=None, can_render=False):
        self.Logs = Logs(self)

        self.config = {
            'env': {
                'can_render': can_render,
                'path': 'ALE',
                'game': game,
                'frameskip': 4,
                'repeat_action_probability': 0.25,
                'full_action_space': True,
                'render_mode': render_mode,  # None, human, rgb_array, single_rgb_array...
                'seed': 42
            },
            'hyperparameters': {  # Hyperparameters are external settings defined by the user to control the model training process
                'gamma': 0.99,  # Discount factor for past rewards
                'epsilon': self.Logs.logs['epsilon'] if epsilon == -1 else epsilon,  # Epsilon greedy parameter
                'epsilon_max': 1.0,  # Maximum epsilon greedy parameter
                'epsilon_min': epsilon_min,  # Minimum epsilon greedy parameter
                'max_steps_per_episode': 10000,
                'epsilon_random_frames': 20,
                'epsilon_greedy_frames': 1000000,
                'batch_size': 32,
            },
            'parameters': {  # Parameters are learned values within the model
                'update_target_network': 10000,
                'update_after_actions': 4,
            }
        }
        self.epsilon_interval = (self.config['hyperparameters']['epsilon_max'] - self.config['hyperparameters']['epsilon_min'])  # Rate at which to reduce chance of random action being taken
        self.save_logs = True

        self.env = self.make_env()
        self.env.seed(self.config['env']['seed'])

        self.Model = Model(self, load_model=load_model)
        self.Memory = Memory(self)

    def quit(self):
        self.Logs.write("Stopping the program...")
        self.Logs.write("Saving the model...")
        self.Model.save_model()
        self.Logs.write("Saving the logs...")
        self.Logs.save()
        self.Logs.write("Quitting the program...")

    def make_env(self):
        return gym.make(
            f"{self.config['env']['path']}/{self.config['env']['game']}",
            frameskip=self.config['env']['frameskip'],
            repeat_action_probability=self.config['env']['repeat_action_probability'],
            full_action_space=self.config['env']['full_action_space'],
            render_mode=self.config['env']['render_mode']
        )

    def train(self):
        self.Logs.write("Starting the training...")

        while True:
            state, info = np.array(self.env.reset(), dtype=object)

            for timestep in range(1, self.config['hyperparameters']['max_steps_per_episode']):
                if self.config['env']['can_render'] == True:
                    env.render()  # Adding this line would show the attempts of the agent in a pop up window but it slows down the training

                # Use epsilon-greedy for exploration
                if self.Logs.logs['frame_count_training'] < self.config['hyperparameters']['epsilon_random_frames'] or self.config['hyperparameters']['epsilon'] > np.random.rand(1)[0]:
                    # Take random action
                    action = np.random.choice(list(GAMES[self.config['env']['game']]['inputs'].keys()))
                    self.Logs.logs['action_type'] = 'random'
                else:
                    # Predict action Q-values from environment state
                    try:
                        state_tensor = tf.convert_to_tensor(state)
                        state_tensor = tf.expand_dims(state_tensor, 0)
                        action_probs = self.Model.model(state_tensor, training=False)
                        # Take best action
                        action = tf.argmax(action_probs[0]).numpy()
                        self.Logs.logs['action_type'] = 'predicted'
                    except Exception as e:
                        self.Logs.error("While predicting action", e)
                        print(action_probs[0])
                        action = np.random.choice(list(GAMES[self.config['env']['game']]['inputs'].keys()))
                        self.Logs.logs['action_type'] = 'random'

                self.Logs.logs['action'] = action

                # Decay probability of taking random action
                self.config['hyperparameters']['epsilon'] -= self.epsilon_interval / self.config['hyperparameters']['epsilon_greedy_frames']
                self.config['hyperparameters']['epsilon'] = max(self.config['hyperparameters']['epsilon'], self.config['hyperparameters']['epsilon_min'])
                self.Logs.logs['epsilon'] = self.config['hyperparameters']['epsilon']

                # Apply the sampled action in our environment

                state_next, reward, done, _, _ = self.env.step(self.Model.action_key_to_value(action))
                state_next = np.array(state_next)

                self.Logs.logs['reward'] = reward
                self.Logs.logs['reward_game'] += reward
                self.Logs.logs['reward_training'] += reward
                self.Logs.logs['done'] = bool(done)

                # Save actions and states in replay buffer
                self.Memory.action_history.append(action)
                self.Memory.state_history.append(state)
                self.Memory.state_next_history.append(state_next)
                self.Memory.done_history.append(done)
                self.Memory.rewards_history.append(reward)
                state = state_next

                # Update every fourth frame and once batch size is over 32
                if self.Logs.logs['frame_count_training'] % self.config['parameters']['update_after_actions'] == 0 and len(self.Memory.done_history) > self.config['hyperparameters']['batch_size']:

                    # Get indices of samples for replay buffers
                    indices = np.random.choice(range(len(self.Memory.done_history)), size=self.config['hyperparameters']['batch_size'])

                    # Using list comprehension to sample from replay buffer
                    state_sample = np.array([self.Memory.state_history[i] for i in indices], dtype=object)
                    state_next_sample = np.array([self.Memory.state_next_history[i] for i in indices])
                    rewards_sample = [self.Memory.rewards_history[i] for i in indices]
                    action_sample = [self.Memory.action_history[i] for i in indices]
                    done_sample = tf.convert_to_tensor([float(self.Memory.done_history[i]) for i in indices])

                    # Build the updated Q-values for the sampled future states
                    # Use the target model for stability
                    future_rewards = self.Model.model_target.predict(state_next_sample, verbose=0)
                    # Q value = reward + discount factor * expected future reward
                    updated_q_values = rewards_sample + self.config['hyperparameters']['gamma'] * tf.reduce_max(future_rewards, axis=1)

                    # If final frame set the last value to -1
                    updated_q_values = updated_q_values * (1 - done_sample) - done_sample

                    # Create a mask so we only calculate loss on the updated Q-values
                    masks = tf.one_hot(action_sample, len(GAMES[self.config['env']['game']]['inputs']))

                    with tf.GradientTape() as tape:
                        # Train the model on the states and updated Q-values
                        for i in range(len(state_sample)):
                            if state_sample[i].shape != self.Model.config['parameters']['input_shape']:
                                state_sample[i] = np.zeros(self.Model.config['parameters']['input_shape'])
                        state_sample = [sample for sample in state_sample if sample.shape == self.Model.config['parameters']['input_shape']]

                        state_sample = tf.convert_to_tensor(state_sample, dtype=tf.float32)
                        q_values = self.Model.model(state_sample)

                        # Apply the masks to the Q-values to get the Q-value for action taken
                        q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                        # Calculate loss between new Q-value and old Q-value
                        loss = self.Model.loss_function(updated_q_values, q_action)

                    # Backpropagation
                    grads = tape.gradient(loss, self.Model.model.trainable_variables)
                    self.Model.optimizer.apply_gradients(zip(grads, self.Model.model.trainable_variables))

                if self.Logs.logs['frame_count_training'] % self.config['parameters']['update_target_network'] == 0:
                    # update the the target network with new weights
                    self.Model.update()
                    self.Model.save_model()

                # Limit the state and reward history
                if len(self.Memory.rewards_history) > self.Memory.max_memory_length:
                    del self.Memory.rewards_history[:1]
                    del self.Memory.state_history[:1]
                    del self.Memory.state_next_history[:1]
                    del self.Memory.action_history[:1]
                    del self.Memory.done_history[:1]

                # Appel de la fonction pour enregistrer les logs dans le fichier CSV
                self.Logs.push()

                if self.save_logs:
                    self.Logs.save()

                if done:
                    break

            self.Logs.logs['game_number_total'] += 1
            self.Logs.logs['game_number_training'] += 1
            self.Logs.logs['frame_count_game'] = 0
            self.Logs.logs['executed_time_game'] = 0
            self.Logs.logs['reward_game'] = 0

            # Update running reward to check condition for solving
            self.Memory.episode_reward_history.append(self.Logs.logs['reward_game'])
            if len(self.Memory.episode_reward_history) > 100:
                del self.Memory.episode_reward_history[:1]
            running_reward = np.mean(self.Memory.episode_reward_history)

            if running_reward > 40:  # Condition to consider the task solved
                self.Logs.write(f"Solved at episode {self.Logs.logs['game_number_training']}!")
                break

    def play(self):
        self.Logs.write("Starting the game...")

        state, info = np.array(self.env.reset(), dtype=object)

        while True:
            if self.config['env']['can_render'] == True:
                self.env.render()

            # Predict action using the trained model
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = self.Model.model(state_tensor, training=False)
            action = tf.argmax(action_probs[0]).numpy()

            self.Logs.logs['action'] = action
            self.Logs.logs['action_type'] = 'predicted'

            # Apply the action to the environment
            print(self.Model.action_key_to_value(action))
            state_next, reward, done, _, _ = self.env.step(self.Model.action_key_to_value(action))
            state_next = np.array(state_next)

            self.Logs.logs['reward'] = reward
            self.Logs.logs['done'] = bool(done)

            state = state_next

            # Appel de la fonction pour enregistrer les logs dans le fichier CSV
            # self.Logs.push()

            # if self.save_logs:
            #     self.Logs.save()

            if done:
                self.Logs.write("Game over!")
                break

        self.Logs.write("Game finished.")

        # Close the environment if rendering
        if self.config['env']['can_render']:
            self.env.close()