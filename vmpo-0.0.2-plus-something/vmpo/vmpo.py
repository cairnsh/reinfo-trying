import warnings
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as F

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn

from .buffers import RolloutDistBuffer

SMALLEST_ETA = 1e-8
SMALLEST_ALPHA = 1e-8
EPSILON_ETA = 0.2
EPSILON_ALPHA = 0.1
LAGRANGIAN_LEARNING_RATE = 1e-3
POPART_VARIANCE_MIN = 1e-8
POPART_AVERAGING_BETA = 1e-1

class VMPO(OnPolicyAlgorithm):
    """
    V-MPO (Value Maximum a Posteriori Policy Optimization)

    The default optimizer for VMPO is SGD!

    Paper: https://arxiv.org/abs/1909.12238v1
    Code: The starting point for this code was the implementation of PPO in Stable Baselines 3
        (https://github.com/DLR-RM/stable-baselines3)
        The acknowledgements from that are:
            This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
            https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
            Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = False,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        epsilon_eta = EPSILON_ETA,
        epsilon_alpha = EPSILON_ALPHA,
    ):

        # the default policy for this is SGD
        if policy_kwargs is None:
            policy_kwargs = {}
        policy_kwargs.setdefault("optimizer_class", th.optim.SGD)

        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert (
                buffer_size > 1
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self.epsilon_eta = epsilon_eta
        self.epsilon_alpha = epsilon_alpha

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        # but replace the rollout buffer with a RolloutDistBuffer

        self.rollout_buffer = RolloutDistBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

        def param_with_initial_value(z):
            return th.nn.Parameter(th.Tensor([z]).to(self.device))

        self.policy.eta = param_with_initial_value(SMALLEST_ETA)
        self.policy.alpha = param_with_initial_value(SMALLEST_ALPHA)
        self.popart_initializer()
        self.popart_forward_affine_transform_of_value_network()


    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        policy_losses, temperature_losses, kl_losses, value_losses = [], [], [], []
        clip_fractions = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                ### POPART HAPPENS HERE

                # if the values are so high that they blot out the sun, then we shall PopArt in the shade
                # (see Learning values across many orders of magnitude, 1602.07714)

                # don't use sample variance
                with th.no_grad():
                    returns_sq, returns_mu = th.var_mean(rollout_data.returns, unbiased=False)
                    returns_sq = returns_sq.item()
                    returns_mu = returns_mu.item()

                self.popart_update(returns_sq, returns_mu)
                self.popart_inverse_affine_transform_of_value_network()

                ### END OF POPART

                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                log_dist = self.policy.get_distribution(rollout_data.observations).distribution.logits
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # throw out half of the data
                full = advantages.shape[0]
                half = full//2
                advantages_h, indices_h = th.topk(advantages, half)
                values_h = values[indices_h]
                log_prob_h = log_prob[indices_h]
                #entropy = entropy[indices_h]

                eta = self.policy.eta

                advantages_offset = th.max(advantages_h)
                advantages_h -= advantages_offset

                e_advantages = th.exp(advantages_h / eta)
                sum_of_e_advantages = th.sum(e_advantages)
                psi = e_advantages / sum_of_e_advantages

                policy_loss = -th.dot(psi, log_prob_h)

                temperature_loss = eta * (self.epsilon_eta + advantages_offset/eta + th.log(sum_of_e_advantages / half))
                #if np.random.random() < 0.001:
                    #print(eta, values_h)

                # now we have to get the kl loss between the old action distribution and the new one
                # (using the whole batch of states)
                # but we can approximate it like this??

                old_dist = rollout_data.distributions
                approximatekl = th.sum(th.exp(old_dist) * (old_dist - log_dist))
                alpha = self.policy.alpha

                kl_loss = th.mean(
                    alpha * (self.epsilon_alpha - approximatekl.detach()) + alpha.detach() * approximatekl
                )

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target


                popart_sigma2, popart_mu = self.popart_updated_params
                popart_return = (rollout_data.returns - popart_mu) / np.sqrt(popart_sigma2)
                value_loss = F.mse_loss(popart_return, values_pred)

                loss = policy_loss + temperature_loss + kl_loss + self.vf_coef * value_loss

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

                with th.no_grad():
                    # the lagrangian parameters aren't in the optimizer...
                    # i believe the mot juste is "lol"
                    # so we'll update them here using sgd
                    def lagrangian_parameter_update(z, lower_limit):
                        z[0] -= LAGRANGIAN_LEARNING_RATE * z._grad[0]
                        if z[0] < lower_limit:
                            z[0] = lower_limit
                    lagrangian_parameter_update(alpha, SMALLEST_ALPHA)
                    lagrangian_parameter_update(eta, SMALLEST_ETA)

                self.popart_forward_affine_transform_of_value_network()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        #self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        #self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        #self.logger.record("train/value_loss", np.mean(value_losses))
        #self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        #self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        #self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def get_value_layer(self):
        return self.policy.value_net

    def popart_update(self, sigma2, mu):
        # the concept of popart is that you maintain moving averages of the mean and std dev of the value targets, then
        # instantly rescale and shift the final layer of the network from the old average to the new average without going through gradients
        # the value loss is supposed to be calculated with the unscaled network against inverse-scaled value targets, so that its parameters stay around their original initialization
        x2, x = self.popart_moving_average
        x2 += POPART_AVERAGING_BETA * (sigma2 + mu*mu - x2)
        x += POPART_AVERAGING_BETA * (mu - x)
        
        self.popart_moving_average = [x2, x]
        sigma2 = x2 - x*x
        mu = x
        sigma2 = max(sigma2, POPART_VARIANCE_MIN)
        self.popart_updated_params = [sigma2, mu]

    def popart_inverse_affine_transform_of_value_network(self):
        # do the opposite transformation of forward_affine_transform
        old_sigma2, old_mu = self.popart_params
        layer = self.get_value_layer()
        with th.no_grad():
            layer.bias[:] -= old_mu
            layer.weight[:] /= np.sqrt(old_sigma2)
            layer.bias[:] /= np.sqrt(old_sigma2)

        self.popart_params = None

    def popart_forward_affine_transform_of_value_network(self):
        # add scale and bias to the value network
        sigma2, mu = self.popart_updated_params
        layer = self.get_value_layer()
        with th.no_grad():
            layer.bias[:] *= np.sqrt(sigma2)
            layer.weight[:] *= np.sqrt(sigma2)
            layer.bias[:] += mu
        self.popart_params = [sigma2, mu]

    def popart_initializer(self):
        self.popart_updated_params = [1, 0] # variance, mean
        self.popart_moving_average = [1, 0] # x2, x

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "VMPO",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "PPO":

        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )
