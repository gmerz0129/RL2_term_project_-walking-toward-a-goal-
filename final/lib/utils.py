import argparse
import sys
from pathlib import Path
import torch
import cv2
import gymnasium as gym
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

import custom_envs
from lib.wrappers import GoalRewardWrapper, StandingResetWrapper


def parse_args_ppo() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    # Device & Environment
    parser.add_argument("--cuda", action="store_true", default=True, help="Use CUDA")
    parser.add_argument("--env", default="HumanoidHole-v0", help="Environment ID")
    parser.add_argument("--n-envs", type=int, default=16, help="Parallel environments")
    
    # Training Params
    parser.add_argument("--n-epochs", type=int, default=500, help="Training epochs")
    parser.add_argument("--n-steps", type=int, default=2048, help="Steps per epoch")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--train-iters", type=int, default=10, help="Iterations per epoch")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-ratio", type=float, default=0.2, help="PPO clip ratio")
    parser.add_argument("--ent-coef", type=float, default=0.005, help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function coefficient")
    parser.add_argument("--target-kl", type=float, default=0.05, help="Target KL (Early Stopping)")
    parser.add_argument("--reward-scale", type=float, default=0.01, help="Reward scaling")
    
    # Goal Parameters
    parser.add_argument("--goal-x", type=float, default=7.5, help="Goal X")
    parser.add_argument("--goal-y", type=float, default=2.5, help="Goal Y")
    parser.add_argument("--goal-radius", type=float, default=0.7, help="Goal radius")
    
    # Logging
    parser.add_argument("--render-epoch", type=int, default=10, help="Render frequency")
    parser.add_argument("--save-epoch", type=int, default=10, help="Save frequency")
    parser.add_argument("--resume", type=str, default=None, help="Resume path")

    # Feature Flags: Rewards
    parser.add_argument("--enable-base", action="store_true", help="Enable Humanoid Default Reward")
    parser.add_argument("--enable-progress", action="store_true", help="Enable Progress Reward")
    parser.add_argument("--enable-velocity", action="store_true", help="Enable Velocity Reward")
    parser.add_argument("--enable-heading", action="store_true", help="Enable Heading Reward")
    parser.add_argument("--enable-upright", action="store_true", help="Enable Upright Reward")
    parser.add_argument("--enable-distance", action="store_true", help="Enable Distance Reward")
    parser.add_argument("--enable-medium-bonus", action="store_true", help="Enable Medium Bonus")

    # Feature Flags: Costs
    parser.add_argument("--enable-hole-penalty", action="store_true", help="Enable Hole Fall Penalty")
    parser.add_argument("--enable-action-cost", action="store_true", help="Enable Action Penalty")
    parser.add_argument("--enable-joint-vel-cost", action="store_true", help="Enable Joint Velocity Penalty")
    parser.add_argument("--enable-contact-cost", action="store_true", help="Enable Contact Penalty")
    parser.add_argument("--enable-lateral-cost", action="store_true", help="Enable Lateral Deviation Penalty")
    parser.add_argument("--enable-smooth-cost", action="store_true", help="Enable Action Smoothness Penalty")
    
    # Other Flags
    parser.add_argument("--use-standing-reset", action="store_true", help="Use StandingResetWrapper")
    parser.add_argument("--use-obs-norm", action="store_true", help="Enable Observation Normalization")
    
    return parser.parse_args()


def make_env(
    env_id: str,
    goal_xy: tuple,
    goal_radius: float,
    reward_scaling: float = 1.0,
    render: bool = False,
    render_mode: str = None,
    # Rewards
    enable_base: bool = False,
    enable_progress: bool = False,
    enable_velocity: bool = False,
    enable_heading: bool = False,
    enable_upright: bool = False,
    enable_distance: bool = False,
    enable_medium_bonus: bool = False,
    # Costs
    enable_hole_penalty: bool = False,
    enable_action_cost: bool = False,
    enable_joint_vel_cost: bool = False,
    enable_contact_cost: bool = False,
    enable_lateral_cost: bool = False,
    enable_smooth_cost: bool = False,
    # Wrappers
    use_standing_reset: bool = False,
):
    if render_mode is None:
        render_mode = "rgb_array" if render else None
    
    env = gym.make(
        env_id,
        render_mode=render_mode,
        terminate_when_unhealthy=True,
        forward_reward_weight=0.0,
        reset_noise_scale=0.0,
        exclude_current_positions_from_observation=False,
        healthy_z_range=(0.9, 2.0),
    )
    
    if use_standing_reset:
        env = StandingResetWrapper(env, pose_noise=1e-4, vel_noise=1e-5, settle_steps=4)
    
    env = GoalRewardWrapper(
        env,
        goal_xy=goal_xy,
        goal_radius=goal_radius,
        reward_scale=reward_scaling,
        # Feature Flags
        use_base_reward=enable_base,
        use_progress_reward=enable_progress,
        use_velocity_reward=enable_velocity,
        use_heading_reward=enable_heading,
        use_upright_reward=enable_upright,
        use_distance_reward=enable_distance,
        use_medium_bonus_reward=enable_medium_bonus,
        
        use_hole_penalty=enable_hole_penalty,
        use_action_penalty=enable_action_cost,
        use_joint_vel_penalty=enable_joint_vel_cost,
        use_contact_penalty=enable_contact_cost,
        use_lateral_penalty=enable_lateral_cost,
        use_smooth_penalty=enable_smooth_cost,
    )
    
    return env


def log_video(
    env,
    agent,
    device: torch.device,
    video_path: str,
    max_steps: int = 1000,
):
    # 에피소드 비디오 저장
    agent.eval()
    frames = []
    
    obs, _ = env.reset()
    done = False
    step = 0
    
    while not done and step < max_steps:
        frame = env.render()
        if frame is not None and len(frame) > 0:
            frames.append(frame)
        
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action, _, _, _ = agent.get_action_and_value(obs_tensor)
        
        obs, _, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
        done = terminated or truncated
        step += 1
        
        if info.get("is_success", False):
            break
    
    if frames:
        height, width = frames[0].shape[:2]
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))
        for frame in frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()
        print(f"Video saved: {video_path}")
