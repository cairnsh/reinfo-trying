import gym
from stable_baselines3 import PPO
from several_algorithms import SeveralAlgorithms
from vecenv import MatrixGameVecEnv

env = MatrixGameVecEnv(2, 2, {
    (0, 0): 3,
    (0, 1): 1,
    (1, 0): 4,
    (1, 1): 0
})

model = SeveralAlgorithms([
    PPO.load("cooperate_plz-%d" % idx, env)
    for idx in range(2)
], env)

print("Get the players to play for a few rounds.")

name_of = {
    0: "Cooperate",
    1: "Defect"
}

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    print(obs)
    obs, rewards, dones, info = env.step(action)
    for j in range(2):
        print("Player %d vs. %d:" % (j, obs[j, 0]))
        print("    %s vs. %s" % (name_of[obs[j, -2]], name_of[obs[j, -1]]))
        print("    Reward: %f" % rewards[j])
        print()
