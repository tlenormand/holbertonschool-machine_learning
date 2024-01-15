#!/usr/bin/env python3
# https://github.com/chengxi600/RLStuff/blob/master/Q%20Learning/Atari_DQN.ipynb

import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Lambda
from tqdm import tqdm_notebook
from tensorflow import keras
import tensorflow as tf
import datetime
from collections import deque
import random
import cv2
import stable_baselines3.common.atari_wrappers as atari_wrappers


#Initialize environment. v4 means no action repeat
env = gym.make('PongNoFrameskip-v4')

#wraps env with these preprocessing options:
#values will be scaled at training time to save memory
"""Atari 2600 preprocessings. 
    This class follows the guidelines in 
    Machado et al. (2018), "Revisiting the Arcade Learning Environment: 
    Evaluation Protocols and Open Problems for General Agents".
    Specifically:
    * NoopReset: obtain initial state by taking random number of no-ops on reset. 
    * Frame skipping: 4 by default
    * Max-pooling: most recent two observations
    * Termination signal when a life is lost: turned off by default. Not recommended by Machado et al. (2018).
    * Resize to a square image: 84x84 by default
    * Grayscale observation: optional
    * Scale observation: optional"""
env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=4, 
                                      screen_size=84, terminal_on_life_loss=False, 
                                      grayscale_obs=True, grayscale_newaxis=False, scale_obs=False)

env = gym.wrappers.FrameStack(env, 4)

env = atari_wrappers.ClipRewardEnv(env)


#actions in this environment
env.unwrapped.get_action_meanings()

#number of frames to run
NUM_FRAMES = 1000000

#number of episodes to run
NUM_EPISODES = 50

#max iterations per run
MAX_ITERATIONS = 1000000

#epsilon for choosing action
eps = 1

#minimum eps
eps_min = 0.1

#eps linear decay for first 10% of run
eps_linear_decay = (eps-eps_min)/(NUM_FRAMES/5)

#discount factor for future utility
discount_factor = 0.99

#batch size for exp replay
batch_size = 32

#max memory stored for exp replay
MAX_MEMORY = int(NUM_FRAMES/10)

#initial population of memory using random policy
INIT_MEMORY = int(NUM_FRAMES/20)

#update interval to use target network
TARGET_C = int(NUM_FRAMES/1000)

#keep scores
scores = []
frames = 0

#iterate through 10 playthroughs
for _ in tqdm_notebook(range(1)):
    
    #reset env
    env.reset()
    done = False
    score = 0
    
    #while game is not over
    while not done:
        #render env
        env.render()
        frames += 1
        
        #execute random action
        _, reward, done, _ = env.step(env.action_space.sample())
        
        #track score
        score += reward
        
    #append to score list
    scores.append(score)
env.close()

def epsilon_greedy(eps, model, env, state):
    ''' Returns an action using epsilon greedy strategy
    Args:
    - eps (int): chance for random action
    - model (Model): Keras model used to choose best action
    - env (EnvSpec): Gym environment
    
    Returns:
    - (int): index of best action
    '''
    #exploration
    if np.random.random() < eps:
        #exploration
        action = np.random.randint(0, env.action_space.n)
        return action
    else:
        #exploitation
        #use expand_dims here to add a dimension for input layer
        q_vals = model.predict(state)
        action = np.argmax(q_vals)
        return action
def experience_replay(memory, model, target_model, discount_factor, batch_size):
    ''' Fits the model with minibatch of states from memory
    Args:
    - memory (Array): array of environment transitions
    - model (Model): Keras model to be fit
    - target_model (Model): Keras model to get target Q val
    - discount_factor (float): discount factor for future utility
    - batch_size (int): size of minibatch
    
    Returns: None
    '''
    
    #if memory is less than batch size, return nothing
    if len(memory) < batch_size:
        return
    else:
        states = []
        targets = []
        
        #sample a batch
        minibatch = random.sample(memory, batch_size)
        
        #iterate through bastch
        for state, action, reward, new_state, done in minibatch:
            #scale states to be [0,1]. We only scale before fitting cuz storing uint8 is cheaper
            state = state/255
            new_state = new_state/255

            target = reward
            
            #if game not over, target q val includes discounted future utility
            #we use a cloned model to predict here for stability. Model is changed every C frames
            #we use the online model to choose best action to deal with overestimation error (Double-Q learning)
            if not done:
                best_future_action = np.argmax(model.predict(new_state))
                target = reward + discount_factor * target_model.predict(new_state)[0][best_future_action]
            
            #get current actions vector
            target_vector = model.predict(state)[0]
            
            #update current action q val with target q val
            target_vector[action] = target
            
            #add to states
            states.append(state)
            
            #add to targets
            targets.append(target_vector)
            
        #fit model
        model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)
        
def ncwh_to_nwhc(tensor):
    '''Converts tensor from NCWH to NWHC
    Args:
    - tensor (4D Array): NCWH tensor
    
    Returns:
    - (4D Array): tensor in NWHC format
    '''
    return tf.transpose(tensor, [0, 2, 3, 1])

    #I use lambda layer so I can convert NCWH to NWHC since CPU training doesn't support NCWH
model = Sequential(
    [
        Lambda(ncwh_to_nwhc, output_shape=(84, 84, 4), input_shape=(4, 84, 84)),
        Conv2D(16, kernel_size=(8, 8), strides=4, activation="relu", input_shape=(4, 84, 84)),
        Conv2D(32, kernel_size=(4, 4), strides=2, activation="relu"),
        Flatten(),
        Dense(256, activation="relu"),
        Dense(env.action_space.n, activation="linear"),
    ]
)

rms = tf.keras.optimizers.RMSprop(learning_rate=0.00025, momentum=0.95, epsilon=0.01)
model.compile(loss=tf.keras.losses.Huber(), optimizer=rms)
model.summary()
target_model = Sequential(
    [
        Lambda(ncwh_to_nwhc, output_shape=(84, 84, 4), input_shape=(4, 84, 84)),
        Conv2D(16, kernel_size=(8, 8), strides=4, activation="relu", input_shape=(4, 84, 84)),
        Conv2D(32, kernel_size=(4, 4), strides=2, activation="relu"),
        Flatten(),
        Dense(256, activation="relu"),
        Dense(env.action_space.n, activation="linear"),
    ]
)
#Prefill memory with INIT_MEMORY frames

#init memory using deque to only store MAX_MEMORY
memory = deque(maxlen=MAX_MEMORY)

#progress bar
pbar = tqdm_notebook(total=INIT_MEMORY)

#playthrough game until memory is prefilled
while len(memory) < INIT_MEMORY:
    
    #reset env
    state = env.reset()

    done = False
    
    #playthrough
    while not done:
        
        #random action
        action = env.action_space.sample()
        
        #execute action
        new_state, reward, done, info = env.step(action)
        
        #add transition to memory
        memory.append([np.expand_dims(state, axis=0), action, reward, np.expand_dims(new_state, axis=0), done])
        
        #progress bar
        pbar.update(1)
        
        #update state
        state = new_state
        
#close progress bar
pbar.close()
#init scores
scores = []

#init total_frames
total_frames = 0

#init num_updates
num_updates = 0

#init fitness history
fit_hist = {'loss': []}
pbar = tqdm_notebook(total=50)

#run frames
while total_frames < NUM_FRAMES:
        
    state = env.reset()
    done = False
    score = 0
    frames = 0
            
    #playing through this round
    for frame in range(MAX_ITERATIONS):
        env.render()
        
        frames += 1
        
        #epsilon greedy choose action
        action = epsilon_greedy(eps, model, env, np.expand_dims(state, axis=0))
        
        
        #execute action
        new_state, reward, done, info = env.step(action)
        
        #track score
        score += reward
        
        #memorize
        memory.append([np.expand_dims(state, axis=0), action, reward, np.expand_dims(new_state, axis=0), done])
        
        #exp replay
        experience_replay(memory, model, model, discount_factor, batch_size)
        
        #clone target network every C frames
        num_updates += batch_size
        
        if num_updates > TARGET_C:
            num_updates = 0
            target_model.set_weights(model.get_weights())
            
            #save memory and model
            np.save('memory', memory)
            model.save('tmp_model')
            
        
        #update state
        state = new_state
        
        #decay epsilon
        eps -= eps_linear_decay
        eps = max(eps, eps_min)
        
        if done:
            break
    
    scores.append(score)
    total_frames += frames
    pbar.update(1)
    
pbar.close()
        
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns

sns.set()

plt.plot(scores)
plt.ylabel('score')
plt.xlabel('episodes')
plt.title('CartPole')

reg = LinearRegression().fit(np.arange(len(scores)).reshape(-1, 1), np.array(scores).reshape(-1, 1))
y_pred = reg.predict(np.arange(len(scores)).reshape(-1, 1))
plt.plot(y_pred)
plt.show()

done = False
score = 0
state = env.reset()
q_hist = []
scores = []

for _ in range(10):
    state = env.reset()
    done = False
    score = 0
    while not done:
        env.render()
        state = np.array(state)/255
        action = np.argmax(model.predict(np.expand_dims(state, axis=0)))
        q_hist.append(model.predict(np.expand_dims(state, axis=0)).mean())
        new_state, reward, done, info = env.step(action)
        score += reward
        state = new_state
    scores.append(score)
env.close()
model = keras.models.load_model('tmp_model', custom_objects={'tf':tf})

model.predict(np.expand_dims(env.reset(), axis=0))
