#!/usr/bin/env python3

import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
import os
import csv

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

LOGS = {
    'game_number_total': 0,
    'game_number_training': 0,
    'frame_count_total': 0,
    'frame_count_training': 0,
    'frame_count_game': 0,
    'executed_time_total': 0,
    'executed_time_training': 0,
    'executed_time_game': 0,
    'training_number': 0,
    'epsilon': 0,
    'action': 0,
    'action_type': 0,
    'reward': 0,
    'reward_game': 0,
    'reward_training': 0
}

# numéro de la partie
GAME_NUMBER_TOTAL = 0
# nombre de frames total
FRAME_COUNT_TOTAL = 0
# nombre de frames de la partie
FRAME_COUNT_GAME = 0
# nombre de frames depuis le début de cet entraînement
FRAME_COUNT_TRAINING = 0
# temps total d'exécution
EXECUTED_TIME_TOTAL = 0
# temps d'exécution de la partie
EXECUTED_TIME_GAME = 0
# temps d'exécution depuis le début de cet entraînement
EXECUTED_TIME_TRAINING = 0
# numéro d'entrainement
TRAINING_NUMBER = 0
# epsilon
epsilon = 0.8393377000053883  # Epsilon greedy parameter
# action
# action_type
# reward
reward = 0
# reward de la partie
REWARD_GAME = 0
# reward depuis le début de cet entraînement
REWARD_TRAINING = 0

# global variables
start_time = time.time()
intermediate_time = start_time
# Configuration paramaters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
    epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken
batch_size = 32  # Size of batch taken from replay buffer
max_steps_per_episode = 10000
SHAPE=(210, 160, 3)
load_model = True
save_logs = True

# Use the Baseline Atari environment because of Deepmind helper functions
env = gym.make(
    "ALE/Breakout-v5",
    frameskip=4,
    repeat_action_probability=0.25,
    full_action_space=True,
    render_mode=None,  # human, rgb_array, single_rgb_array...
)
# gym.make(
#     self.path_game + self.game,
#     obs_type='grayscale',
#     frameskip=4,
#     repeat_action_probability=0.25,
#     full_action_space=True,
#     render_mode=render_mode,  # human, rgb_array, single_rgb_array...
# )
# Warp the frames, grey scale, stake four frame and scale to smaller ratio
# env = wrap_deepmind(env, frame_stack=True, scale=True)
env.seed(seed)

num_actions = 4

def save_logs(logs):
    global intermediate_time

    executed_time = time.time() - intermediate_time
    intermediate_time = time.time()

    LOGS['frame_count_total'] += 1
    LOGS['frame_count_game'] += 1
    LOGS['frame_count_training'] += 1
    LOGS['executed_time_total'] += executed_time
    LOGS['executed_time_game'] += executed_time
    LOGS['executed_time_training'] += executed_time
    LOGS['reward_game'] += reward
    LOGS['reward_training'] += reward

    print(LOGS)
    # Vérifiez si le dossier 'logs' existe, sinon, créez-le
    if not os.path.exists('logs'):
        os.mkdir('logs')

    # Spécifiez le nom du fichier CSV
    csv_file = 'logs/logs.csv'
    csv_file2 = 'logs/logs2.csv'

    # Vérifiez si le fichier CSV existe déjà
    file_exists = os.path.isfile(csv_file)
    file_exists2 = os.path.isfile(csv_file2)

    # Ouvrez le fichier CSV en mode écriture
    with open(csv_file, 'a', newline='') as csvfile:
        fieldnames = logs.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Si le fichier n'existe pas, écrivez l'en-tête
        if not file_exists:
            writer.writeheader()

        # Écrivez les données dans le fichier CSV
        writer.writerow(logs)

    # Ouvrez le fichier CSV en mode écriture
    with open(csv_file2, 'a', newline='') as csvfile:
        fieldnames = LOGS.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Si le fichier n'existe pas, écrivez l'en-tête
        if not file_exists2:
            writer.writeheader()

        # Écrivez les données dans le fichier CSV
        writer.writerow(LOGS)

def init_logs():
    def format_logs(last_line):
        if not last_line:
            return {}

        try:
            LOGS['game_number_total'] = int(last_line['game_number_total']) + 1
        except:
            LOGS['game_number_total'] = 0

        LOGS['game_number_training'] = 0

        try:
            LOGS['frame_count_total'] = int(last_line['frame_count_total'])
        except:
            LOGS['frame_count_total'] = 0

        LOGS['frame_count_game'] = 0
        LOGS['frame_count_training'] = 0

        try:
            LOGS['executed_time_total'] = float(last_line['executed_time_total'])
        except:
            LOGS['executed_time_total'] = 0

        LOGS['executed_time_game'] = 0
        LOGS['executed_time_training'] = 0

        try:
            LOGS['training_number'] = int(last_line['training_number'])
        except:
            LOGS['training_number'] = 0

        LOGS['epsilon'] = epsilon
        LOGS['action'] = 0
        LOGS['action_type'] = ''
        LOGS['reward'] = 0
        LOGS['reward_game'] = 0
        LOGS['reward_training'] = 0

        return LOGS

    if not os.path.exists('logs'):
        os.mkdir('logs')

    csv_file = 'logs/logs2.csv'

    if os.path.isfile(csv_file):
        with open(csv_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            last_line = None
            for row in reader:
                last_line = row

            return format_logs(last_line)
    else:
        return {}

init_logs()

def create_q_model():
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=SHAPE)

    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(num_actions, activation="linear")(layer5)

    return keras.Model(inputs=inputs, outputs=action)


# The first model makes the predictions for Q-values which are used to
# make a action.
model = create_q_model() if not load_model else keras.models.load_model('models/gym_policy.h5')
# Build a target model for the prediction of future rewards.
# The weights of a target model get updated every 10000 steps thus when the
# loss between the Q-values is calculated the target Q-value is stable.
model_target = create_q_model() if not load_model else keras.models.load_model('models/gym_target_policy.h5')

# In the Deepmind paper they use RMSProp however then Adam optimizer
# improves training time
optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

# Experience replay buffers
action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []
episode_reward_history = []
running_reward = 0
episode_count = 0
frame_count = 0
# Number of frames to take random action and observe output
epsilon_random_frames = 20
# Number of frames for exploration
epsilon_greedy_frames = 1000000.0
# Maximum replay length
# Note: The Deepmind paper suggests 1000000 however this causes memory issues
max_memory_length = 100000
# Train the model after 4 actions
update_after_actions = 4
# How often to update the target network
update_target_network = 1000
# Using huber loss for stability
loss_function = keras.losses.Huber()

def action_to_key(action):
    keys_list = list(BREAKOUT_INPUTS.keys())
    values_list = list(BREAKOUT_INPUTS.values())

    return keys_list[values_list.index(action)]

def action_to_value(action):
    keys_list = list(BREAKOUT_INPUTS.keys())
    values_list = list(BREAKOUT_INPUTS.values())

    return values_list[action]

while True:  # Run until solved
    state = np.array(env.reset())
    episode_reward = 0

    for timestep in range(1, max_steps_per_episode):
        # env.render()  # Adding this line would show the attempts
        # of the agent in a pop up window.
        frame_count += 1

        # Use epsilon-greedy for exploration
        if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
            # Take random action
            action = np.random.choice(list(BREAKOUT_INPUTS.keys()))
            action_type = 'random'
        else:
            # Predict action Q-values
            # From environment state
            try:
                state_tensor = tf.convert_to_tensor(state)
                state_tensor = tf.expand_dims(state_tensor, 0)
                action_probs = model(state_tensor, training=False)
                # Take best action
                action = tf.argmax(action_probs[0]).numpy()
                action_type = 'predicted'
            except Exception as e:
                print("Error: ", e)
                action = np.random.choice(list(BREAKOUT_INPUTS.keys()))
                action_type = 'random'

        LOGS['action'] = action

        # Decay probability of taking random action
        epsilon -= epsilon_interval / epsilon_greedy_frames
        epsilon = max(epsilon, epsilon_min)

        LOGS['epsilon'] = epsilon

        # Apply the sampled action in our environment
        state_next, reward, done, _, _ = env.step(action_to_value(action))
        state_next = np.array(state_next)

        episode_reward += reward

        LOGS['reward'] = reward
        LOGS['reward_game'] = REWARD_GAME + reward
        LOGS['reward_training'] = REWARD_TRAINING + reward

        # Save actions and states in replay buffer
        action_history.append(action)
        state_history.append(state)
        state_next_history.append(state_next)
        done_history.append(done)
        rewards_history.append(reward)
        state = state_next

        # Exemple d'utilisation
        logs = {
            'executed time': str(np.round(time.time() - start_time, 2)),
            'frame_count': frame_count,
            'epsilon': epsilon,
            'action': action,
            'action_type': action_type,
            'reward': reward,
            'episode_reward': episode_reward
        }

        # Appel de la fonction pour enregistrer les logs dans le fichier CSV
        if save_logs:
            save_logs(logs)

        # Update every fourth frame and once batch size is over 32
        if frame_count % update_after_actions == 0 and len(done_history) > batch_size:

            # Get indices of samples for replay buffers
            indices = np.random.choice(range(len(done_history)), size=batch_size)

            # Using list comprehension to sample from replay buffer
            state_sample = np.array([state_history[i] for i in indices])
            state_next_sample = np.array([state_next_history[i] for i in indices])
            rewards_sample = [rewards_history[i] for i in indices]
            action_sample = [action_history[i] for i in indices]
            done_sample = tf.convert_to_tensor(
                [float(done_history[i]) for i in indices]
            )

            # Build the updated Q-values for the sampled future states
            # Use the target model for stability
            future_rewards = model_target.predict(state_next_sample, verbose=0)
            # Q value = reward + discount factor * expected future reward
            updated_q_values = rewards_sample + gamma * tf.reduce_max(
                future_rewards, axis=1
            )

            # If final frame set the last value to -1
            updated_q_values = updated_q_values * (1 - done_sample) - done_sample

            # Create a mask so we only calculate loss on the updated Q-values
            masks = tf.one_hot(action_sample, num_actions)

            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                for i in range(len(state_sample)):
                    if state_sample[i].shape != SHAPE:
                        state_sample[i] = np.zeros(SHAPE)
                state_sample = [sample for sample in state_sample if sample.shape == SHAPE]

                state_sample = tf.convert_to_tensor(state_sample, dtype=tf.float32)
                q_values = model(state_sample)

                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                loss = loss_function(updated_q_values, q_action)

            # Backpropagation
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if frame_count % update_target_network == 0:
            # update the the target network with new weights
            model_target.set_weights(model.get_weights())
            # Log details
            template = "running reward: {:.2f} at episode {}, frame count {}"
            print(template.format(running_reward, episode_count, frame_count))

            # save model
            model.save('models/gym_policy.h5')
            # save target model
            model_target.save('models/gym_target_policy.h5')


        # Limit the state and reward history
        if len(rewards_history) > max_memory_length:
            del rewards_history[:1]
            del state_history[:1]
            del state_next_history[:1]
            del action_history[:1]
            del done_history[:1]

        if done:
            break

    FRAME_COUNT_GAME = 0
    EXECUTED_TIME_GAME = 0
    REWARD_GAME = 0
    LOGS['game_number_training'] += 1
    LOGS['frame_count_game'] = 0
    LOGS['executed_time_game'] = 0
    LOGS['reward_game'] = 0

    # Update running reward to check condition for solving
    episode_reward_history.append(episode_reward)
    if len(episode_reward_history) > 100:
        del episode_reward_history[:1]
    running_reward = np.mean(episode_reward_history)

    episode_count += 1

    if running_reward > 40:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        break
