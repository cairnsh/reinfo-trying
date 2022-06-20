import torch
from vmpo import VMPO
from several_algorithms import SeveralAlgorithms
from vecenv import MatrixGameVecEnv

env = MatrixGameVecEnv(2, 6, {
    (0, 0): 0,
    (0, 1): 0.01,
    (1, 0): 0.02,
    (1, 1): 0
})

class RecorderCallback:
    def __init__(self, filename):
        self.recordf = open(filename, "w")
    def special_record_games(self, opponents, actions):
        self.recordf.write("%s %s\n" % (opponents, actions))
class pseudo:
    def __init__(self):
        # set some of the local variables to match the real algorithms
        # THIS IS A HACK:

        # this has to be the same as the n_steps in the other algorithms
        self.n_steps = 2048
        # we have to keep track of this for SeveralAlgorithms to work
        self.num_timesteps = 0
        # this has to be the same as the device in the other algorithms
        # (on my system this is "cuda" but it might have to be "cpu" on others)
        self.device = torch.device("cuda")

        # END OF HACK
        # ALL THE REST OF THE CODE IS COMPLETELY PRINCIPLED

        self.action_space = env.action_space

        # real algorithms would have a rollout buffer and a policy object
        # so we have to make a fake version

        class fake_rollout_buffer:
            def reset(self):
                pass
            def add(self, last_obs, action, reward, last_done, value, log_prob):
                pass
            def compute_returns_and_advantage(self, last_values, dones):
                pass

        self.rollout_buffer = fake_rollout_buffer()

        class fake_policy:
            def __init__(self, parent):
                self.parent = parent
            def set_training_mode(self, *_, **__):
                pass
            def __call__(self, obs):
                # play tit for tat
                # get the last action
                opponent = obs[0, -1]

                action = self.parent.decide(opponent)
                return (
                    torch.tensor([action]),
                    torch.tensor([0]), # value (just set to whatever)
                    torch.tensor([0]), # logprob (just set to whatever)
                )
            def predict_values(self, __):
                pass

        self.policy = fake_policy(self)
    def _setup_learn(self, timesteps, *_, **__):
        return timesteps, None
    def predict(self, obs):
        raise NotImplemented
    def train(self):
        pass
    def save(self, path, include, exclude):
        fo = open(path, "w")
        fo.write(self.save_data())
        fo.close()
    def save_data(self):
        raise NotImplemented # implement me in subclass
    def decide(self, last_opponent_action):
        raise NotImplemented # implement me in subclass
    @staticmethod
    def load(path):
        raise NotImplemented # implement me in subclass

class tit_for_tat(pseudo):
    def decide(self, opponent):
        if opponent == 2:
            # first round
            return 0
        return opponent
    def save_data(self):
        return "Tit for tat\n"

class all_cooperate(pseudo):
    def decide(self, opponent):
        return 0
    def save_data(self):
        return "All cooperate\n"

if True:
    # Instantiate the agent
    model = SeveralAlgorithms([
        VMPO('MlpPolicy', env, batch_size=128, learning_rate=1e-4, gamma=0.9, verbose=1)
        for j in range(6)
    ], env)
    # Train the agent
    import codeit
    codeit.codeit(l=locals())
    model.learn(callback=RecorderCallback("recording.txt"), total_timesteps=int(120000))
    # Save the agent
    model.save("battle_of_the_sexes")
