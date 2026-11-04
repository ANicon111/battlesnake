"""Tasks: implement a one-step Q update, then connect the agent/environment loop.

The supplied runner creates a zero-initialized Q-table and manages episodes,
epsilon decay, logging, plots, checkpoint files and videos. You implement learning.
No labelled actions, neural network or pre-trained policy is provided.
"""

import numpy as np
from step_1_q_state import encode_state
from step_2_q_choose_action import choose_action


def update_q(q, state, action, reward, next_state, terminated, alpha, gamma):
    """Modify ONLY q[state][action] in place. Do not make a copy of q or return anything."""
    target = reward if terminated else reward + gamma * np.max(q[next_state])
    q[state][action] += alpha * (target - q[state][action])


def train_episode(env, q, epsilon, rng, seed, alpha, gamma):
    """Use env.reset(seed=seed), then repeatedly call env.step(action)."""
    observation, info = env.reset(seed=seed)
    state = encode_state(observation)
    episode_reward = 0.0

    while True:
        action = choose_action(q[state], epsilon, rng)
        next_observation, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        if terminated or truncated:
            next_state = None if terminated else encode_state(next_observation)
            update_q(q, state, action, reward, next_state, terminated, alpha, gamma)
            break
        else:
            next_state = encode_state(next_observation)
            update_q(q, state, action, reward, next_state, False, alpha, gamma)
            state = next_state

    return dict(
        reward=episode_reward,
        steps=env.turn,
        apples=info["apples"],
        terminated=terminated,
        truncated=truncated,
        reason=info["reason"],
    )


if __name__ == "__main__":
    from supplied_experiment import train_cli

    train_cli(encode_state, choose_action, train_episode)
