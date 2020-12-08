import time
import datetime
import os
import numpy as np


class Logger():

    def __init__(self, logging_directory):

        # Create directory to save data
        timestamp = time.time()
        timestamp_value = datetime.datetime.fromtimestamp(timestamp)
        self.base_directory = os.path.join(logging_directory, timestamp_value.strftime('%Y-%m-%d.%H:%M:%S'))
        print('Create data logging session: %s' % self.base_directory)
        self.transitions_directory = os.path.join(self.base_directory, 'transitions')
        self.models_directory = os.path.join(self.base_directory, 'models')

        if not os.path.exists(self.transitions_directory):
            os.makedirs(self.transitions_directory)
        if not os.path.exists(self.models_directory):
            os.makedirs(self.models_directory)

    # Save reward
    def save_reward_value(self, reward_value):
        np.savetxt(os.path.join(self.transitions_directory, 'reward.txt'), reward_value, delimiter='\n')

