from stable_baselines3.common.vec_env import VecEnv
import numpy as np, scipy as sp
from gym import spaces

HISTORYLENGTH = 4

EPISODELENGTH = 100

class Test:
    def __init__(self):
        self.action_space = spaces.Discrete(2,)
        self.observation_space = spaces.MultiDiscrete([2,] * 200)
        self.metadata = {'render_modes': ["human"]}
        pass

class MatrixGameVecEnv(VecEnv):
    def __init__(self, n_actions, n_players, matrix):
        self.n_players = n_players
        self.empty_history_entry = n_actions
        self.num_envs = n_players
        self.rewards = matrix

        self.action_space = spaces.Discrete(n_actions)

        self.metadata = {'render_modes': ["human"]}

        observations = [self.n_players] # other player's id

        observations += [
            n_actions + 1, # your action (n_actions means it's blank)
            n_actions + 1, # other action (n_actions means it's blank)
        ] * HISTORYLENGTH

        self.observation_space = spaces.MultiDiscrete(observations)

        self.n_observations = len(observations)

        self.observations_per_player = [np.zeros(self.n_observations, dtype=np.int32) for _ in range(n_players)]

        self._reset_reward_records()

        self.random = np.random.default_rng()

    def _randomize_matching(self):
        p = self.random.permutation(self.n_players)
        matching = [None] * self.n_players
        for i in range(0, self.n_players, 2):
            a, b = p[i], p[i+1]
            matching[a] = b
            matching[b] = a
        self.matching = matching

    def _reset_observations_using_matching(self):
        for i in range(self.n_players):
            observation = self.observations_per_player[i]

            # the first entry is the other player's id
            observation[0] = self.matching[i]

            # the other entries are the history
            # they're all blank to start with.
            observation[1:] = self.empty_history_entry

    def _update_observation_with_history(self, i, your_action, other_action):
        observation = self.observations_per_player[i]

        # move the history down by one step
        observation[1:-2] = observation[3:]

        # fill in the new
        observation[-2] = your_action
        observation[-1] = other_action

    def _summarize_this_episode_for_a_player(self, i):
        return {
            "episode": {
                "r": self.rewards_this_episode[i],
                "l": EPISODELENGTH
            }
        }

    def _reset_reward_records(self): # blah
        self.rewards_this_episode = [0] * self.n_players

    def reset(self):
        self._randomize_matching()
        self._reset_observations_using_matching()
        self.time = 0
        self._reset_reward_records()
        return np.stack(self.observations_per_player)

    def step(self, action):
        other_action = [action[self.matching[i]] for i in range(self.n_players)]
        reward = [self.rewards[action[i], other_action[i]] for i in range(self.n_players)]
        for i in range(self.n_players):
            self.rewards_this_episode[i] += reward[i]
        for i in range(self.n_players):
            self._update_observation_with_history(i, action[i], other_action[i])

        # This copies the observations, which is important because we might reset them
        observations = np.stack(self.observations_per_player)

        # The done signal comes at the start of an episode...?
        done = [self.time == 0] * self.n_players

        self.time += 1
        if self.time == EPISODELENGTH:
            info = [self._summarize_this_episode_for_a_player(i) for i in range(self.n_players)]

            self.reset()
        else:
            info = [{} for _ in range(self.n_players)]

        return observations, np.array(reward), np.array(done), info

    # VecEnv is an abstract class, so we have to define all of these

    def close(self):
        pass

    def env_is_wrapped(self, wrapper_class, indices = None):
        return False

    def env_method(self, method_name, *method_args, indices = None, **method_kwargs):
        pass

    def get_attr(self, attr_name, indices = None):
        pass
    
    def set_attr(self, attr_name, value, indices = None):
        pass

    def get_images(self):
        pass

    def render(self, mode = "human"):
        raise NotImplemented

    def step_async(self):
        raise NotImplemented

    def step_wait(self):
        raise NotImplemented

    def seed(self, seed=None):
        self.random = np.random.default_rng(seed)
