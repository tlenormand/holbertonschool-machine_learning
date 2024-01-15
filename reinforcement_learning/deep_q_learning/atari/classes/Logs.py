#!/usr/bin/env python3

import copy
import csv
import os
import random
import sys
import time

import numpy as np
from tabulate import tabulate


class Logs():
    def __init__(self, Atari):
        self.config = {
            'logs_path': 'logs/',
            'logs_name': 'logs.csv',
            'rules': {
                'period_display': 100,  # Display logs every X frames
                'period_save': 100,  # Save logs every X frames
            },
            'get_from_previous_logs': [  # Get these values from the last line of the logs file to continue the logs
                'program_execution_number',
                'game_number_total',
                'frame_count_total',
                'executed_time_total',
                'epsilon',
            ]
        }
        self.logs = {
            'program_execution_number': int(0),  # How many times the program has been executed (./main.py)
            'game_number_total': int(0),  # How many games have been played since the beginning
            'game_number_training': int(0),  # How many games have been played since the last program execution (./main.py)
            'frame_count_total': int(0),  # How many frames have been played since the beginning
            'frame_count_training': int(0),  # How many frames have been played since the last program execution (./main.py)
            'frame_count_game': int(0),  # How many frames have been played since the last game
            'executed_time_total': float(0),  # How many seconds have been executed since the beginning
            'executed_time_training': float(0),  # How many seconds have been executed since the last program execution (./main.py)
            'executed_time_game': float(0),  # How many seconds have been executed since the last game
            'epsilon': float(0),  # Epsilon value
            'action': int(0),  # Action value
            'action_type': str(""),  # Action key
            'reward': int(0),  # Reward value
            'reward_game': int(0),  # Reward value since the last game
            'reward_training': int(0),  # Reward value since the last program execution (./main.py)
            'new_model': False,  # If a new model has been created
            'done': False,  # If the game is done
        }
        self.logs_history = []
        self.start_time = time.time()
        self.intermediate_time = self.start_time

        self.Atari = Atari
        self.create_logs_file()
        self.init_logs()

    def create_logs_file(self):
        if not os.path.exists(self.config['logs_path']):
            os.makedirs(self.config['logs_path'])

        if not os.path.isfile(self.config['logs_path'] + self.config['logs_name']):
            with open(self.config['logs_path'] + self.config['logs_name'], 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=list(self.logs.keys()))
                writer.writeheader()

    def init_logs(self):
        if not os.path.exists(self.config['logs_path']):
            os.mkdir(self.config['logs_path'])

        csv_file = self.config['logs_path'] + self.config['logs_name']
        if os.path.isfile(csv_file):
            with open(csv_file, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                last_line = None
                for row in reader:
                    last_line = row

                self.format_last_line(last_line)

    def display(self):
        if self.logs['frame_count_total'] % self.config['rules']['period_display'] != 0 and not self.logs['done'] and not self.logs['new_model'] and not self.logs['reward']:
            return

        # Create a list of key-value pairs from self.logs, rounding the values to 6 decimal places if they are of float type
        table_data = [[key, value if not isinstance(value, float) else round(value, 6)] for key, value in self.logs.items()]
        # Transpose the list of key-value pairs to display the data in columns, not rows
        table_data = list(map(list, zip(*table_data)))
        # Use the 'tabulate' library to format the data as a pipe-separated table
        table = tabulate(table_data, tablefmt='pipe')
        self.write(table)

    def write(self, string):
        sys.stdout.write(string + '\n')
        sys.stdout.flush()

    def error(self, info, e):
        red = "\033[1;31m"
        reset = "\033[0;0m"
        self.write(f"{red}ERROR{reset}: {info}\n{e}")

    def push(self):
        executed_time = time.time() - self.intermediate_time
        self.intermediate_time = time.time()

        self.logs['frame_count_total'] += 1
        self.logs['frame_count_game'] += 1
        self.logs['frame_count_training'] += 1
        self.logs['executed_time_total'] += executed_time
        self.logs['executed_time_game'] += executed_time
        self.logs['executed_time_training'] += executed_time

        self.logs_history.append(self.logs.copy())

        self.display()

        # reset unique values
        self.logs['new_model'] = False
        self.logs['done'] = False

    def save(self):
        if self.logs['frame_count_total'] % self.config['rules']['period_save'] != 0:
            return

        with open(self.config['logs_path'] + self.config['logs_name'], 'a') as csvfile:
            for log in self.logs_history:
                writer = csv.DictWriter(csvfile, fieldnames=list(log.keys()))
                formatted_log = self.format_logs(log)
                if type(formatted_log) == list:
                    for formatted_log in formatted_log:
                        writer.writerow(formatted_log)
                else:
                    writer.writerow(formatted_log)

        self.logs_history = []

    def format_logs(self, logs=None):
        if logs is None:
            logs = self.logs
        
        if type(logs) == list:
            formatted_logs = []
            for log in logs:
                formatted_logs.append(self.format_logs(log))
            return formatted_logs

        formatted_logs = {}
        for key, value in logs.items():
            if type(value) == bool:
                formatted_logs[key] = int(value)
            elif type(value) == float:
                formatted_logs[key] = float(np.format_float_positional(value, precision=6, trim='k'))
            elif type(value) == int:
                formatted_logs[key] = int(value)
            else:
                formatted_logs[key] = value

        return formatted_logs

    def format_last_line(self, last_line):
        if not last_line:
            return

        for key, value in last_line.items():
            if key in self.config['get_from_previous_logs']:
                expected_type = type(self.logs[key])
                self.logs[key] = expected_type(value)
            elif type(self.logs[key]) == bool:
                self.logs[key] = False
            elif type(self.logs[key]) == float:
                self.logs[key] = 0
            elif type(self.logs[key]) == int:
                self.logs[key] = 0
            else:
                self.logs[key] = str("")

            if key == 'program_execution_number':
                self.logs[key] += 1
