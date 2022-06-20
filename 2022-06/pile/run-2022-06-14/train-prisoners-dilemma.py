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

class RecorderCallback:
    def __init__(self, filename):
        self.recordf = open(filename, "w")
    def special_record_games(self, opponents, actions):
        self.recordf.write("%s %s\n" % (opponents, actions))
if True:
    # Instantiate the agent
    model = SeveralAlgorithms([
        PPO('MlpPolicy', env, batch_size=1024, learning_rate=1e-3, verbose=1)
        for j in range(4)
    ], env)
    # Train the agent
    model.learn(callback=RecorderCallback("recording.txt"), total_timesteps=int(1000000))
    # Save the agent
    model.save("prisoners_dilemma")
