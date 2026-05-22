import argparse
import sys
import time
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import custom_envs  
from lib.agent_ppo import PPOAgent
from lib.utils import make_env

def parse_args_test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument("--single", action="store_true", help="Run single episode with detailed log")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    
    # Goal Params
    parser.add_argument("--goal-x", type=float, default=7.5)
    parser.add_argument("--goal-y", type=float, default=2.5)
    parser.add_argument("--goal-radius", type=float, default=0.7)
    
    # Feature Flags
    parser.add_argument("--enable-base", action="store_true")
    parser.add_argument("--enable-progress", action="store_true")
    parser.add_argument("--enable-velocity", action="store_true")
    parser.add_argument("--enable-heading", action="store_true")
    parser.add_argument("--enable-upright", action="store_true")
    parser.add_argument("--enable-distance", action="store_true")
    parser.add_argument("--enable-medium-bonus", action="store_true")
    
    parser.add_argument("--enable-hole-penalty", action="store_true")
    parser.add_argument("--enable-action-cost", action="store_true")
    parser.add_argument("--enable-joint-vel-cost", action="store_true")
    parser.add_argument("--enable-contact-cost", action="store_true")
    parser.add_argument("--enable-lateral-cost", action="store_true")
    parser.add_argument("--enable-smooth-cost", action="store_true")
    
    parser.add_argument("--use-standing-reset", action="store_true")
    parser.add_argument("--use-obs-norm", action="store_true")
    
    return parser.parse_args()

def run_test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    render_mode = "human" if not args.no_render else None
    
    env = make_env(
        env_id="HumanoidHole-v2",
        goal_xy=(args.goal_x, args.goal_y),
        goal_radius=args.goal_radius,
        render=False,
        render_mode=render_mode,
        
        enable_base=args.enable_base,
        enable_progress=args.enable_progress,
        enable_velocity=args.enable_velocity,
        enable_heading=args.enable_heading,
        enable_upright=args.enable_upright,
        enable_distance=args.enable_distance,
        enable_medium_bonus=args.enable_medium_bonus,
        
        enable_hole_penalty=args.enable_hole_penalty,
        enable_action_cost=args.enable_action_cost,
        enable_joint_vel_cost=args.enable_joint_vel_cost,
        enable_contact_cost=args.enable_contact_cost,
        enable_lateral_cost=args.enable_lateral_cost,
        enable_smooth_cost=args.enable_smooth_cost,
        use_standing_reset=args.use_standing_reset
    )
    
    env = gym.wrappers.ClipAction(env)
    if args.use_obs_norm:
        env = gym.wrappers.NormalizeObservation(env)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    agent = PPOAgent(obs_dim, act_dim).to(device)
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if "agent_state_dict" in checkpoint:
        agent.load_state_dict(checkpoint["agent_state_dict"])
    else:
        agent.load_state_dict(checkpoint)
    
    if args.use_obs_norm:
        if hasattr(env, "obs_rms") and "obs_mean" in checkpoint:
            env.obs_rms.mean = checkpoint["obs_mean"]
            env.obs_rms.var = checkpoint["obs_var"]
            env.obs_rms.count = checkpoint["obs_count"]
            env.obs_rms.update = lambda x: None
            print("Obs normalization loaded.")
        else:
            print("Obs norm not found!")

    agent.eval()
    print(f"\nStart Testing ({args.episodes} episodes)...")
    
    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        
        while not done and steps < args.max_steps:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                if args.stochastic:
                    action, _, _, _ = agent.get_action_and_value(obs_tensor)
                else:
                    action, _ = agent.forward(obs_tensor)
                
                action = action.squeeze(0).cpu().numpy()
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            
            if not args.no_render:
                env.render()
                time.sleep(0.01)
                
            if args.single:
                print(f"Step {steps:4d} | Reward: {reward:6.2f} | Dist: {info.get('distance_to_goal',0):.2f}")

        status = "SUCCESS" if info.get("is_success") else "FAILED"
        if info.get("is_hole_fall"): status = "HOLE FALL"
        
        dist = info.get("distance_to_goal", 0.0)
        print(f"Ep {ep+1}: {status} | Steps: {steps} | Reward: {total_reward:.2f} | Final Dist: {dist:.2f}m")

    env.close()

if __name__ == "__main__":
    args = parse_args_test()
    run_test(args)
