import gym
from stable_baselines3 import PPO
import numpy as np
from vecenv import MatrixGameVecEnv

env = MatrixGameVecEnv(2, 4, {
    (0, 0): 3,
    (0, 1): 0,
    (1, 0): 4,
    (1, 1): 1
})

def getcord():
    while True:
        cord = input("Play C or D? ").strip()
        if cord == "c" or cord == "C":
            return 0
        if cord == "d" or cord == "D":
            return 1
        print("I didn't recognize that as C or D.")

class pseudo_observations:
    def __init__(self,i):
        self.observation = np.zeros(1 + 2*4)
        self.observation[0] = i # other player
        self.observation[1:] = 2 # blank

    def add(self, my, other):
        self.observation[1:-2] = self.observation[3:]
        self.observation[-2] = my
        self.observation[-1] = other

    def get(self):
        return self.observation

cordname = {0: 'C', 1: 'D'}

algorithm = [None] * 2 + [PPO.load("prisoners_dilemma-%d" % idx) for idx in range(2, 4)]

class play_against:
    def __init__(self, who_to_be, who_play_against):
        self.algorithm = algorithm[who_to_be]
        self.observations = pseudo_observations(who_play_against)
        self.action = None
        pass

    def play(self, choice):
        action = self.step()
        self.add(choice)
        return action

    def step(self):
        self.action, _states = self.algorithm.predict(self.observations.get())
        return self.action

    def add(self, choice):
        self.observations.add(self.action, choice)
        self.action = None

    def get(self):
        if self.action is None:
            raise Exception("there's no stored action right now!")
        action = self.action
        self.action = None
        return action

def play_head_to_head(a, b):
    A = play_against(a, b)
    B = play_against(b, a)
    while True:
        aa = A.step()
        bb = B.step()
        A.add(bb)
        B.add(aa)
        yield aa, bb

def play_tit_for_tat(a, b):
    bb = 0
    A = play_against(a, b)
    while True:
        aa = A.play(bb)
        yield aa, bb
        bb = aa

def stat(generator):
    ca = 0
    cb = 0
    TURNS = 2000
    for j in range(TURNS):
        a, b = next(generator)
        if a == 0:
            ca += 1
        if b == 0:
            cb += 1
    print("Cooperation: %f%% vs. %f%%" % (ca/TURNS*100, cb/TURNS*100))

for z in range(2, 4):
    print("Playing against %d, and..." % z)
    for i in range(4):
        print("Playing tit for tat, but as player %d" %i)
        stat(play_tit_for_tat(z, i))
    print()

for z in range(2, 4):
    print("Playing against %d, and..." % z)
    for i in range(2, 4):
        print("Playing as %d, and:" % i)
        stat(play_head_to_head(z, i))
