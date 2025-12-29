import gym
import random
import numpy as np

# slippery ajanın envirenent üzerinde kaygan bir zeminde hareket ettiğini varsayan parametre,
# yani false diyerek direkt istediğimiz staten diğer state gider
#render mode bizim göresellleştirmemiz için gerekli
envirenment = gym.make("FrozenLake-v1", is_slippery = False, render_mode = "ansi")
envirenment.reset() # başlangıca geçiyoruz

# kaç tane state var onu bulur -> 4*4 olduğundan 16 tane
nb_states = envirenment.observation_space.n
nb_actions = envirenment.action_space.n # action sayısı
qtable = np.zeros((nb_states,nb_actions))

action = envirenment.action_space.sample() # rastgele bir hareketi seçer
'''
sol -> 0
aşağı -> 1
sağ -> 2
yukarı -> 3
'''

new_state, reward, done, info, _ = envirenment.step(action) # actiona göre hareket eder, 1 adım atar

#%% 

import gym
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# slippery ajanın envirenent üzerinde kaygan bir zeminde hareket ettiğini varsayan parametre,
# yani false diyerek direkt istediğimiz staten diğer state gider
#render mode bizim göresellleştirmemiz için gerekli
envirenment = gym.make("FrozenLake-v1", is_slippery = False, render_mode = "ansi")
envirenment.reset() # başlangıca geçiyoruz

# kaç tane state var onu bulur -> 4*4 olduğundan 16 tane
nb_states = envirenment.observation_space.n
nb_actions = envirenment.action_space.n # action sayısı
qtable = np.zeros((nb_states,nb_actions))

#parametreler
episodes = 10000 # oyunun kaç kere oynanacağı
alpha = 0.5 # learning rate
gamma = 0.9 # discount factor

outcomes = []

#training
for _ in tqdm(range(episodes)):
    state, _ = envirenment.reset() # başlama noktasını döndürür
    done = False
    outcomes.append("Failure")
    while not done: # ajan başarılı olana kadaraction seç ve oyna
    # qtable ajanımızın beyni
        if np.max(qtable[state]) > 0: # eğer bir şey öğrendiyse
            action = np.argmax(qtable[state]) # 
        else:
            action = envirenment.action_space.sample()

        new_state, reward, done, info, _ = envirenment.step(action)
        #update qtable
        qtable[state, action] = qtable[state, action] + alpha*(reward + gamma * np.max(qtable[new_state]) - qtable[state,action]) 
        
        state = new_state
        if reward:
            outcomes[-1] = "Success"
    
print("Q-table after training")
print(qtable)   

plt.figure(figsize=(10, 5))
plt.bar(range(episodes),outcomes)
plt.show()

# test

episodes = 100
nb_success = 0

for _ in tqdm(range(episodes)):
    state, _ = envirenment.reset()
    done = False
    
    while not done:
        if np.max(qtable[state]) > 0 :
            action = np.argmax(qtable[state])
        else:
            action = envirenment.action_space.sample()

        new_state, reward, done, info, _ = envirenment.step(action)
        state = new_state
        nb_success += reward

print("Success rate:", 100*nb_success / episodes)






