# Code credits: Sentdex (https://www.youtube.com/watch?v=yMk_XtIEzH8&t=267s)

import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")

LEARNING_RATE = 0.1
DISCOUNT = 0.95  # how important we find future actions
EPISODES = 10000

SHOW_EVERY = 100

DISCRETE_OS_SIZE = [20]*len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n])) #makes a 20*20*3 q_table with negative values

successful_action_sequences = []
successful_action_sequences_lefts = []
successful_action_sequences_stays = []
successful_action_sequences_rights = []

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))


for episode in range(EPISODES):

    discrete_state = get_discrete_state(env.reset())
    done = False
    action_sequence = []

    if episode % SHOW_EVERY == 0:
        #print(episode)
        render = True
    else:
        render = False

    while not done:
        action = np.argmax(q_table[discrete_state])  # start with a random action
        action_sequence.append(action)

        new_state, reward, done, _ = env.step(action)

        new_discrete_state = get_discrete_state(new_state)

        # if render:
        #     env.render()

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]

            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            q_table[discrete_state + (action, )] = new_q  # updating the qtable after the action has been taken

        elif new_state[0] >= env.goal_position:
            #print(f"We made it on {episode}")
            q_table[discrete_state + (action, )] = 0

            successful_action_sequences.append(action_sequence)
            #print(f"LEFT: {action_sequence.count(0)}, STAY: {action_sequence.count(1)}, RIGHT: {action_sequence.count(2)}")
            successful_action_sequences_lefts.append(action_sequence.count(0))
            successful_action_sequences_stays.append(action_sequence.count(1))
            successful_action_sequences_rights.append(action_sequence.count(2))

        discrete_state = new_discrete_state


env.close()

p1 = plt.plot(successful_action_sequences_lefts, 'bo', markersize=1)
p2 = plt.plot(successful_action_sequences_stays, 'go', markersize=1)
p3 = plt.plot(successful_action_sequences_rights, 'ro', markersize=1)
plt.ylabel("Count")
plt.title("#Actions taken in successful tries")
plt.legend((p1[0], p2[0], p3[0]), ('Left', 'Stay', 'Right'))

plt.show()
