# Code credits: Sentdex (https://www.youtube.com/watch?v=yMk_XtIEzH8&t=267s)

import gym
import numpy as np

env = gym.make("MountainCar-v0")
env.reset()

done = False

DISCRETE_OS_SIZE = [20]*len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))  # makes a 20*20*3 q_table with negative values


LEARNING_RATE = 0.1
DISCOUNT = 0.95  # how important we find future actions
EPISODES = 25000

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))


while not done:
    action = 2
    new_state, reward, done, _ = env.step(action)
    print(get_discrete_state(new_state))
    env.render()

env.close()
