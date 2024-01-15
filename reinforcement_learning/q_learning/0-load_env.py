#!/usr/bin/env python3
import gym

def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    return gym.make('FrozenLake-v1', desc=desc, map_name=map_name, is_slippery=is_slippery, render_mode='human')
