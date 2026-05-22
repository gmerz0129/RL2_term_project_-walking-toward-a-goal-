from __future__ import annotations

from gymnasium.envs.registration import register

from .humanoid_hole_env import HumanoidHoleEnv


# Gymnasium registry에 새로운 환경을 등록
register(
    id="HumanoidHole-v0",
    entry_point="custom_envs.humanoid_hole_env:HumanoidHoleEnv",
    max_episode_steps=1000,
)
