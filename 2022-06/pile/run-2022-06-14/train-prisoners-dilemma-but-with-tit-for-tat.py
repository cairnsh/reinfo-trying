import torch
from stable_baselines3 import PPO
from several_algorithms import SeveralAlgorithms
from vecenv import MatrixGameVecEnv

env = MatrixGameVecEnv(2, 6, {
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
class titfortat:
    def __init__(self):
        self.n_steps = 2048
        self.num_timesteps = 0
        self.device = torch.device("cuda")
        self.action_space = env.action_space
        class roll:
            def reset(self):
                pass
            def add(self, last_obs, action, reward, last_done, value, log_prob):
                pass
            def compute_returns_and_advantage(self, last_values, dones):
                pass
        self.rollout_buffer = roll()
        class poli:
            def __init__(self, parent):
                self.parent = parent
            def set_training_mode(self, *_, **__):
                pass
            def __call__(self, obs):
                opponent = obs[0, -1]
                if opponent == 2:
                    action = 0
                else:
                    action = opponent
                return torch.tensor([action]), torch.tensor([0]), torch.tensor([[0]])
            def predict_values(self, __):
                pass
        self.policy = poli(self)
    def _setup_learn(self, timesteps, *_, **__):
        return timesteps, None
    def predict(self, obs):
        self.num_timesteps += 1
        pass
        raise
    def train(self):
        pass
    def save(self, path, include, exclude):
        pass
    @staticmethod
    def load():
        return titfortat()
if True:
    # Instantiate the agent
    model = SeveralAlgorithms([titfortat() for j in range(2)] + [
        PPO('MlpPolicy', env, batch_size=128, learning_rate=1e-3, verbose=1)
        for j in range(4)
    ], env)
    # Train the agent
    model.learn(callback=RecorderCallback("recording.txt"), total_timesteps=int(120000))
    # Save the agent
    model.save("prisoners_dilemma")
