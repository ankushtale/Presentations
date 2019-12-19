# Code credits: Sentdex (https://www.youtube.com/watch?v=yMk_XtIEzH8&t=267s)

import gym

env = gym.make("MountainCar-v0")
env.reset()

# State would be [position, velocity], where these are variables
# Observation Space is range of values of state variables
# actions is actions that can be taken by an agent at any state

print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space.n)  # 0 -> LEFT, 1 -> STAY, 2 -> RIGHT

# In real life scenarios, we might not even know the observation space, so we'll have to explore the env for a long time to know the bounds

done = False

while not done:
    action = 2
    new_state, reward, done, _ = env.step(action)

    print(new_state)
    print(reward)

    env.render()

env.close()