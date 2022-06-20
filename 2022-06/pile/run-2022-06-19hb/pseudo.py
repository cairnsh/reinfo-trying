import torch
from stable_baselines3.common.distributions import CategoricalDistribution
import numpy
log2 = numpy.log(2)

class pseudo:
    def __init__(self, env):
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
            #def add(self, last_obs, action, reward, last_done, value, log_prob):
            def add(self, *whatever):
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
            def get_distribution(self, obs):
                dist = CategoricalDistribution(1)
                dist.proba_distribution(torch.tensor([[-log2, -log2]]))
                return dist

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
