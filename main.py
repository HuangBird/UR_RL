from vrepEnv import ArmEnv
from pd import PD
from ddpg import DDPG
import numpy as np
import os
from logger import Logger
import time
import datetime


MAX_EPISODES = 1000
MAX_EP_STEPS = 50
ON_TRAIN = True
logging_directory = os.path.abspath('logs')
# logger = Logger(logging_directory)

# Set environment
env = ArmEnv()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound

# Set method: pd/RL
pd = PD()
rl = DDPG(a_dim, s_dim, a_bound)


def runpd():
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_r = 0.
        for j in range(MAX_EP_STEPS):
            a = pd.cal(s, np.array([0, 0, -4, 0, 0]))
            s, r, done, safe = env.step(a)
            if done or j == MAX_EP_STEPS - 1 or safe is False:
                print('Ep: %i | %s | %s | ep_r: %.1f | step: %i' % (i, '---' if not done else 'done',
                                                                    'unsafe' if not safe else 'safe', ep_r, j))
                break


def train():
    """ Training the actor """

    logger = Logger(logging_directory)
    env.start_simulation()
    action_var = a_bound  # Action noise factor
    reward_value = []  # Record the reward value for each episode

    for episode in range(MAX_EPISODES):

        s = env.reset()
        episode_reward = 0.  # Total reward for an episode

        for step in range(MAX_EP_STEPS):

            a = rl.choose_action(s)
            a = np.clip(np.random.normal(a, action_var), -3*a_bound, 3*a_bound)  # Add action noise
            # print('Action: ', a)
            s_, r, done, safety = env.step(a)
            # print(s_)
            rl.store_transition(s, a, r, s_)

            if rl.memory_full:
                rl.learn()  # Start to learn once has fulfilled the memory
                action_var *= .999  # Reduce the action randomness once start to learn

            s = s_
            episode_reward += r
            mean_episode_reward = episode_reward/(step+1) 

            if done or step == MAX_EP_STEPS - 1 or safety == 1:
                print('Episode: %i | %s | %s | MER: %f | Steps: %i | Var: %f'
                      % (episode, 'Not inserted' if not done else 'Inserted', 'Unsafe' if safety == 1 else 'Safe',
                         mean_episode_reward, step, action_var))
                break

        reward_value.append(mean_episode_reward)

    logger.save_reward_value(reward_value)
    rl.save_model(logger.models_directory)


def test():
    models_directory = '/home/pengsheng/Optimal_control_Project/UR_robot/UR5_RL_Assembly-inserted-v1/logs/2020-12-08.13:44:16/models'
    rl.restore_model(models_directory)
    env.start_simulation()
    s = env.reset()
    print('start to move')
    while True:
        print('move to next state')
        a = rl.choose_action(s)
        s, r, done,__ = env.step(a)


if __name__ == '__main__':

    timestamp = time.time()
    timestamp_value = datetime.datetime.fromtimestamp(timestamp)
    print("The program starts at： ", timestamp_value.strftime('%Y-%m-%d.%H:%M:%S'))
    
    if ON_TRAIN:
        train()
    else:
        test()

