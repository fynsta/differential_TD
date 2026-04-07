from typing import Dict

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class MertonInnerState:
    obs: jax.Array
    wealth: jax.Array
    steps: jax.Array


@struct.dataclass
class MertonLogState:
    env_state: MertonInnerState
    episode_returns: jax.Array
    episode_lengths: jax.Array
    returned_episode_returns: jax.Array
    returned_episode_lengths: jax.Array


class MertonVectorEnv:
    """Vectorized Merton portfolio environment with built-in auto-reset and logging."""

    def __init__(
        self,
        batch_size: int,
        dt: float,
        horizon: float,
        mu: float,
        sigma: float,
        risk_free_rate: float,
        initial_wealth: float,
        action_limit: float,
        noise_lvl: float,
    ):
        self.batch_size = int(batch_size)
        self.dt = float(dt)
        self.horizon = float(horizon)
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.risk_free_rate = float(risk_free_rate)
        self.initial_wealth = float(initial_wealth)
        self.action_limit = float(action_limit)
        self.noise_lvl = float(noise_lvl)

        self.action_size = 1
        self.observation_size = 2
        self.episode_length = max(1, int(jnp.ceil(self.horizon / self.dt)))

    def _build_obs(self, wealth: jax.Array, steps: jax.Array) -> jax.Array:
        log_wealth = jnp.log(jnp.maximum(wealth, 1e-8))
        remaining_fraction = jnp.maximum(0.0, 1.0 - steps.astype(jnp.float32) / self.episode_length)
        return jnp.stack([log_wealth, remaining_fraction], axis=-1)

    def _add_obs_noise(self, rng: jax.Array, obs: jax.Array) -> jax.Array:
        if self.noise_lvl <= 0.0:
            return obs
        std_dev = jnp.maximum(1e-6, jnp.abs(obs) * self.noise_lvl)
        return obs + jax.random.normal(rng, shape=obs.shape) * std_dev

    def reset(self, rng: jax.Array) -> MertonLogState:
        wealth = jnp.full((self.batch_size,), self.initial_wealth)
        steps = jnp.zeros((self.batch_size,), dtype=jnp.int32)
        obs = self._build_obs(wealth, steps)
        obs = self._add_obs_noise(rng, obs)

        env_state = MertonInnerState(obs=obs, wealth=wealth, steps=steps)
        zeros = jnp.zeros((self.batch_size,), dtype=jnp.float32)
        return MertonLogState(
            env_state=env_state,
            episode_returns=zeros,
            episode_lengths=zeros,
            returned_episode_returns=zeros,
            returned_episode_lengths=zeros,
        )

    def step(
        self,
        rng: jax.Array,
        state: MertonLogState,
        action: jax.Array,
    ):
        rng_dyn, rng_obs = jax.random.split(rng)

        action = jnp.asarray(action)
        if action.ndim == 1:
            action = action[:, None]
        portfolio_weight = jnp.clip(action[:, 0], -self.action_limit, self.action_limit)

        brownian = jax.random.normal(rng_dyn, shape=(self.batch_size,))

        prev_wealth = state.env_state.wealth
        step_count = state.env_state.steps

        drift = (
            self.risk_free_rate
            + portfolio_weight * (self.mu - self.risk_free_rate)
            - 0.5 * (portfolio_weight * self.sigma) ** 2
        ) * self.dt
        diffusion = portfolio_weight * self.sigma * jnp.sqrt(self.dt) * brownian

        wealth_growth = jnp.exp(drift + diffusion)
        next_wealth = jnp.maximum(1e-8, prev_wealth * wealth_growth)

        reward = jnp.log(next_wealth) - jnp.log(jnp.maximum(prev_wealth, 1e-8))

        next_steps = step_count + 1
        done = (next_steps >= self.episode_length).astype(jnp.float32)

        episode_returns = state.episode_returns + reward
        episode_lengths = state.episode_lengths + 1.0

        returned_episode_returns = (
            state.returned_episode_returns * (1.0 - done)
            + episode_returns * done
        )
        returned_episode_lengths = (
            state.returned_episode_lengths * (1.0 - done)
            + episode_lengths * done
        )

        reset_wealth = jnp.full_like(next_wealth, self.initial_wealth)
        post_reset_wealth = jnp.where(done > 0.0, reset_wealth, next_wealth)
        post_reset_steps = jnp.where(done > 0.0, jnp.zeros_like(next_steps), next_steps)
        next_obs = self._build_obs(post_reset_wealth, post_reset_steps)
        next_obs = self._add_obs_noise(rng_obs, next_obs)

        next_env_state = MertonInnerState(
            obs=next_obs,
            wealth=post_reset_wealth,
            steps=post_reset_steps,
        )
        next_state = MertonLogState(
            env_state=next_env_state,
            episode_returns=episode_returns * (1.0 - done),
            episode_lengths=episode_lengths * (1.0 - done),
            returned_episode_returns=returned_episode_returns,
            returned_episode_lengths=returned_episode_lengths,
        )

        info: Dict[str, jax.Array] = {
            "returned_episode_returns": returned_episode_returns,
            "returned_episode_lengths": returned_episode_lengths,
            "returned_episode": done,
        }

        return next_state, next_obs, reward, done, info
