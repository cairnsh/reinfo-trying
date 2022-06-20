import gym
from stable_baselines3 import PPO
from several_algorithms import SeveralAlgorithms
from vecenv import MatrixGameVecEnv

env = MatrixGameVecEnv(2, 4, {
    (0, 0): 3,
    (0, 1): 0,
    (1, 0): 4,
    (1, 1): 1
})

model = SeveralAlgorithms([
    PPO.load("prisoners_dilemma-%d" % idx, env)
    for idx in range(4)
], env)

import codeit;
codeit.codeit(l=locals())

print("Get the players to play for a few rounds.")

name_of = {
    0: "Cooperate",
    1: "Defect"
}

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    for j in range(4):
        print("Player %d vs. %d:" % (j, obs[j, 0]))
        print("    %s vs. %s" % (name_of[obs[j, -2]], name_of[obs[j, -1]]))
        print("    Reward: %f" % rewards[j])
        print()
