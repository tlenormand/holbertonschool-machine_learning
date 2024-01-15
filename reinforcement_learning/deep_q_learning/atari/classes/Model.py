#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow import keras as K
from global_variables import GAMES


class Model():
    def __init__(self, Atari, load_model=False):
        self.Atari = Atari
        self.config = {
            'inputs': GAMES[Atari.config['env']['game']]['inputs'],
            'load_model': load_model,
            'model_path': 'models/',
            'model_name': 'model',
            'model_target_name': 'model_target',
            'parameters': {
                'input_shape': (210, 160, 3),
                'optimizer': {
                    'type': 'Adam',
                    'learning_rate': 0.00025,
                    'clipnorm': 1.0
                },
                'loss_function': {
                    'type': 'Huber'
                }
            }
        }

        self.optimizer = K.optimizers.Adam(
            learning_rate=self.config['parameters']['optimizer']['learning_rate'],
            clipnorm=self.config['parameters']['optimizer']['clipnorm']
        )
        self.loss_function = K.losses.Huber()

        self.model = self.build_model() if not self.config['load_model'] else self.load_model(self.config['model_name'])
        self.model_target = self.build_model() if not self.config['load_model'] else self.load_model(self.config['model_target_name'])

    def action_value_to_key(self, action):
        keys_list = list(self.config['inputs'].keys())
        values_list = list(self.config['inputs'].values())

        return keys_list[values_list.index(action)]

    def action_key_to_value(self, action):
        keys_list = list(self.config['inputs'].keys())
        values_list = list(self.config['inputs'].values())

        return values_list[action]

    def load_model(self, model_name):
        try:
            return K.models.load_model(self.config['model_path'] + model_name)
        except:
            self.Atari.Logs.write(f"Error: Model not found, a new model will be created for {model_name}")
            return self.build_model()

    def save_model(self):
        self.model.save(self.config['model_path'] + self.config['model_name'])
        self.model_target.save(self.config['model_path'] + self.config['model_target_name'])

        # save as json
        model_json = self.model.to_json()
        with open(self.config['model_path'] + self.config['model_name'] + '.json', 'w') as json_file:
            json_file.write(model_json)
        
        # save weights
        self.model.save_weights(self.config['model_path'] + self.config['model_name'] + '.weights.bin')

    def update(self):
        self.model_target.set_weights(self.model.get_weights())

    def build_model(self):
        self.Atari.Logs.logs['new_model'] = True
        model = K.Sequential([
            K.Input(shape=self.config['parameters']['input_shape']),
            K.layers.Conv2D(32, 8, strides=4, activation='relu'),
            K.layers.Conv2D(64, 4, strides=2, activation='relu'),
            K.layers.Conv2D(64, 3, strides=1, activation='relu'),
            K.layers.Flatten(),
            K.layers.Dense(512, activation='relu'),
            K.layers.Dense(len(self.config['inputs']), activation='linear')
        ])

        self.Atari.Logs.write(str(model.summary()))

        model.compile(
            optimizer=self.optimizer,
            loss=self.loss_function
        )

        return model
