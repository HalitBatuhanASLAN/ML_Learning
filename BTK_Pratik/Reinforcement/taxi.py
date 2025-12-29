import gym
import numpy as np

import random

from tqdm import tqdm

env = gym.make("Taxi-v3",render_mode = "ansi")
env.reset()
print(env.render())

'''
0: güney
1: kuzey
2: doğu
3: batı
4: yolcuyu almak
5: yıcuyu bırakmak

'''

action_space = env.action_space.n
state_space = env.observation_space.n

q_table = np.zeros((state_space,action_space))

alpha = 0.1
gamma = 0.6
epsilon = 0.1

for i in tqdm(range(1,100001)):
    state, _ = env.reset()
    done = False
    
    while not done:
        if random.uniform(0,1) < epsilon: # explore adımı eklendi-> 
            action = env.action_space.sample()
        else: # exploit -> aklını kullanıyor
            action = np.argmax(q_table[state])
    
        next_state, reward, done, info, _ = env.step(action)

        q_table[state,action] = q_table[state,action] + alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state,action])
        
        state = next_state
        
print("Training finished")

# test
total_epoch, total_penalty = 0, 0
episode = 100


for i in tqdm(range(episode)):
    state, _ = env.reset()
    
    epoch, penalties, reward = 0, 0, 0
    
    done = False
    
    while not done:
        action = np.argmax(q_table[state])
        next_state, reward, done, info, _ = env.step(action)
        state = next_state
        
        if reward == -10:
            penalties += 1
        epoch += 1
    
    total_epoch += epoch
    total_penalty += penalties

print("Result for after {} episodes".format(episode))
print("Average timesteps per episode:",total_epoch/episode)
print("Average penalty per episode:",total_penalty/episode)
