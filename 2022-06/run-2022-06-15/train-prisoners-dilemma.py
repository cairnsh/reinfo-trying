import gym
from stable_baselines3 import PPO
from several_algorithms import SeveralAlgorithms
from vecenv import MatrixGameVecEnv

env = MatrixGameVecEnv(2, 2, {
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
    def after_training(self, model, timesteps):
        print(timesteps)

if False:
    # Instantiate the agent
    model = SeveralAlgorithms([
        PPO('MlpPolicy', env, batch_size=128, learning_rate=1e-3, verbose=1)
        for j in range(2)
    ], env)
    # Train the agent
    model.learn(callback=RecorderCallback("recording.txt"), total_timesteps=int(100000))
    # Save the agent
    model.save("prisoner")


if True:
    model = SeveralAlgorithms([
        PPO.load("prisoner-%d.zip" % idx, env, batch_size=128, learning_rate=1e-3, verbose=1)
        for idx in range(2)
    ], env)

    """
    name_of = {
        0: "C",
        1: "D"
    }

    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        print(name_of[obs[0, -2]], name_of[obs[0, -1]])
        """

    model.learn(callback=RecorderCallback("recording.txt"), total_timesteps=int(100000))
    model.save("prisonez")
