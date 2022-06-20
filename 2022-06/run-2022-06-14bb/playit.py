import gym
from stable_baselines3 import PPO
import numpy as np
from vecenv import MatrixGameVecEnv

env = MatrixGameVecEnv(2, 6, {
    (0, 0): 3,
    (0, 1): 0,
    (1, 0): 4,
    (1, 1): 1
})

algorith = PPO.load("prisoners_dilemma-2", env)

def getcord():
    while True:
        cord = input("Play C or D? ").strip()
        if cord == "c" or cord == "C":
            return 0
        if cord == "d" or cord == "D":
            return 1
        print("I didn't recognize that as C or D.")

class pseudo_observations:
    def __init__(self):
        self.observation = np.zeros(1 + 2 * 4)
        self.observation[0] = 0 # other player
        self.observation[1:] = 2 # blank

    def add(self, my, other):
        self.observation[1:-2] = self.observation[3:]
        self.observation[-2] = my
        self.observation[-1] = other

    def get(self):
        return self.observation

obs = pseudo_observations()

cordname = {0: 'C', 1: 'D'}

"""
i = 0

while True:
    action, _states = algorith.predict(obs.get())
    humancord = getcord()
    print("%d: %s vs. %s" % (i, cordname[humancord], cordname[action]))
    obs.add(action, humancord)
    i += 1
    pass
    pass
"""

cooperate = 0
other = 0
for i in range(100):
    action, _states = algorith.predict(obs.get())
    obs.add(action, other)
    other = 1 if np.random.random() < 0.5 else 0
    if action == 0:
        cooperate += 1
print(cooperate)
