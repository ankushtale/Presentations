# Code credits: Sentdex (https://www.youtube.com/watch?v=yMk_XtIEzH8&t=267s)

import gym
import numpy as np

env = gym.make("MountainCar-v0")
env.reset()

#Q_table is a table of C*action_space, so for every state we store what's the q value for a particular action

done = False

DISCRETE_OS_SIZE = [20]*len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n])) #makes a 20*20*3 q_table with negative values

LEARNING_RATE = 0.1
DISCOUNT = 0.95  # how important we find future actions
EPISODES = 25000


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))


discrete_state = get_discrete_state(env.reset())


while not done:
    action = np.argmax([discrete_state])  # start with a random action
    new_state, reward, done, _ = env.step(action)

    new_discrete_state = get_discrete_state(new_state)
    env.render()

    if not done:
        max_future_q = np.max(q_table[new_discrete_state])
        current_q = q_table[discrete_state + (action, )]

        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

        q_table[discrete_state + (action, )] = new_q  # updating the qtable after the action has been taken

    elif new_state[0] >= env.goal_position:
        q_table[discrete_state + (action, )] = 0

    discrete_state = new_discrete_state

env.close()
