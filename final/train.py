import datetime
import sys
import time
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import custom_envs 
from lib.agent_ppo import PPOAgent
from lib.buffer_ppo import PPOBuffer
from lib.utils import parse_args_ppo, make_env, log_video


def save_plots(history, save_path):
    epochs = range(1, len(history["rewards"]) + 1)
    if len(epochs) == 0: return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Metrics', fontsize=16)
    
    axes[0, 0].plot(epochs, history["rewards"], 'b-', linewidth=1.5); axes[0, 0].set_title('Mean Reward'); axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(epochs, history["policy_loss"], 'r-', linewidth=1.5); axes[0, 1].set_title('Policy Loss'); axes[0, 1].grid(True, alpha=0.3)
    axes[0, 2].plot(epochs, history["value_loss"], 'g-', linewidth=1.5); axes[0, 2].set_title('Value Loss'); axes[0, 2].grid(True, alpha=0.3)
    axes[1, 0].plot(epochs, history["entropy"], 'm-', linewidth=1.5); axes[1, 0].set_title('Entropy'); axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].plot(epochs, history["lr"], 'c-', linewidth=1.5); axes[1, 1].set_title('Learning Rate'); axes[1, 1].grid(True, alpha=0.3)
    axes[1, 2].plot(epochs, history["kl"], 'y-', linewidth=1.5); axes[1, 2].set_title('KL Divergence'); axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def train():
    args = parse_args_ppo()
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dirs = {k: Path(__file__).parent / k / timestamp for k in ["videos", "checkpoints"]}
    for p in dirs.values(): p.mkdir(parents=True, exist_ok=True)
    
    # Env Factory 
    def env_factory(render=False):
        return make_env(
            args.env, 
            (args.goal_x, args.goal_y), 
            args.goal_radius, 
            args.reward_scale,
            render=render,
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

    envs = gym.vector.AsyncVectorEnv([lambda: env_factory(render=False) for _ in range(args.n_envs)])
    
    envs = gym.wrappers.vector.RecordEpisodeStatistics(envs)
    envs = gym.wrappers.vector.ClipAction(envs)
    if args.use_obs_norm:
        envs = gym.wrappers.vector.NormalizeObservation(envs)
        print(">> Obs Normalization: ENABLED")
    else:
        print(">> Obs Normalization: DISABLED")
    
    test_env = env_factory(render=True)
    test_env = gym.wrappers.RecordEpisodeStatistics(test_env)
    test_env = gym.wrappers.ClipAction(test_env)
    if args.use_obs_norm:
        test_env = gym.wrappers.NormalizeObservation(test_env)
    
    obs_dim = envs.single_observation_space.shape
    act_dim = envs.single_action_space.shape
    agent = PPOAgent(obs_dim[0], act_dim[0]).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    def lr_lambda(epoch):
        warmup_epochs = 10
        if epoch < warmup_epochs: return float(epoch) / float(max(1, warmup_epochs))
        else:
            T_cur, T_total = epoch - warmup_epochs, args.n_epochs - warmup_epochs
            return 0.5 * (1 + np.cos(np.pi * T_cur / T_total))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    start_epoch, best_reward = 1, -np.inf
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        agent.load_state_dict(ckpt["agent_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        
        if args.use_obs_norm and hasattr(envs, "obs_rms") and "obs_mean" in ckpt:
            envs.obs_rms.mean, envs.obs_rms.var, envs.obs_rms.count = ckpt["obs_mean"], ckpt["obs_var"], ckpt["obs_count"]
            print("Loaded obs norm stats")
        
        start_epoch = ckpt.get("epoch", 1) + 1
        best_reward = ckpt.get("best_mean_reward", -np.inf)
        print(f"Resuming from epoch {start_epoch}")

    buffer = PPOBuffer(obs_dim, act_dim, args.n_steps, args.n_envs, device, args.gamma, args.gae_lambda)
    next_obs, _ = envs.reset()
    next_obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.n_envs, dtype=torch.float32, device=device)
    
    history = {"rewards": [], "policy_loss": [], "value_loss": [], "entropy": [], "kl": [], "lr": []}
    
    print(f"Start training: {args.env} on {device}")
    
    try:
        for epoch in range(start_epoch, args.n_epochs + 1):
            ep_rewards, ep_distances = [], []
            
            for _ in tqdm(range(args.n_steps), desc=f"Epoch {epoch}", leave=False):
                with torch.no_grad():
                    actions, logprobs, _, values = agent.get_action_and_value(next_obs)
                
                real_next_obs, rewards, terminateds, truncateds, infos = envs.step(actions.cpu().numpy())
                dones = np.logical_or(terminateds, truncateds)
                buffer.store(next_obs, actions, torch.tensor(rewards).to(device), values.flatten(), 
                             next_done, torch.tensor(truncateds).to(device), logprobs)
                
                next_obs = torch.tensor(real_next_obs, dtype=torch.float32, device=device)
                next_done = torch.tensor(dones, dtype=torch.float32, device=device)
                
                if "episode" in infos:
                    if "_episode" in infos:
                        for i, has_ep in enumerate(infos["_episode"]):
                            if has_ep: ep_rewards.append(infos["episode"]["r"][i])
                if "distance_to_goal" in infos:
                    ep_distances.extend(infos["distance_to_goal"])

            with torch.no_grad():
                next_val = agent.get_value(next_obs).reshape(1, -1)
                adv, ret = buffer.calculate_advantages(next_val, next_done.reshape(1, -1), torch.zeros_like(next_done).reshape(1, -1))
            
            obs_buf, act_buf, logprob_buf = buffer.get()
            b_obs, b_act, b_logprob = obs_buf.reshape(-1, *obs_dim), act_buf.reshape(-1, *act_dim), logprob_buf.reshape(-1)
            b_adv, b_ret = adv.reshape(-1), ret.reshape(-1)
            b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
            
            epoch_losses, epoch_pl, epoch_vl, epoch_ent, epoch_kl = [], [], [], [], []
            inds = np.arange(args.n_steps * args.n_envs)
            kl_early_stop = False
            
            for _ in range(args.train_iters):
                if kl_early_stop: break
                
                np.random.shuffle(inds)
                for start in range(0, len(inds), args.batch_size):
                    end = start + args.batch_size
                    mb_inds = inds[start:end]
                    
                    _, new_logprob, entropy, new_val = agent.get_action_and_value(b_obs[mb_inds], b_act[mb_inds])
                    ratio = torch.exp(new_logprob - b_logprob[mb_inds])
                    surr1, surr2 = ratio * b_adv[mb_inds], torch.clamp(ratio, 1 - args.clip_ratio, 1 + args.clip_ratio) * b_adv[mb_inds]
                    
                    pg_loss = -torch.min(surr1, surr2).mean()
                    v_loss = 0.5 * ((new_val.view(-1) - b_ret[mb_inds]) ** 2).mean()
                    ent = entropy.mean()
                    loss = pg_loss + args.vf_coef * v_loss - args.ent_coef * ent
                    
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                    optimizer.step()
                    
                    with torch.no_grad():
                        kl = ((b_logprob[mb_inds] - new_logprob) / b_act[mb_inds].size(-1)).mean()
                        epoch_losses.append(loss.item()); epoch_pl.append(pg_loss.item()); epoch_vl.append(v_loss.item())
                        epoch_ent.append(ent.item()); epoch_kl.append(kl.item())
                        
                        if kl.item() > args.target_kl:
                            kl_early_stop = True
                            break
            
            scheduler.step()

            avg_reward = np.mean(ep_rewards) if ep_rewards else 0.0
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            avg_kl = np.mean(epoch_kl) if epoch_kl else 0.0
            avg_dist = np.mean(ep_distances) if ep_distances else 0.0
            current_lr = scheduler.get_last_lr()[0]
            
            history["rewards"].append(avg_reward); history["policy_loss"].append(np.mean(epoch_pl) if epoch_pl else 0.0)
            history["value_loss"].append(np.mean(epoch_vl) if epoch_vl else 0.0); history["entropy"].append(np.mean(epoch_ent) if epoch_ent else 0.0)
            history["kl"].append(avg_kl); history["lr"].append(current_lr)
            
            print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Reward: {avg_reward:.2f} | KL: {avg_kl:.4f} | Dist: {avg_dist:.2f} | LR: {current_lr:.2e}")
            
            rms = envs.obs_rms if args.use_obs_norm and hasattr(envs, "obs_rms") else None
            if rms: print(f"  > Norm Stats: Mean={np.mean(rms.mean):.3f}, Var={np.mean(rms.var):.3f}")

            ckpt_data = {
                "epoch": epoch, "agent_state_dict": agent.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "best_mean_reward": max(best_reward, avg_reward),
                "obs_mean": rms.mean if rms else None, "obs_var": rms.var if rms else None, "obs_count": rms.count if rms else None
            }
            
            if avg_reward > best_reward:
                best_reward = avg_reward
                torch.save(ckpt_data, dirs["checkpoints"] / "best.pt")
                save_plots(history, dirs["checkpoints"] / "training_metrics_best.png")
            
            if epoch % args.save_epoch == 0:
                torch.save(ckpt_data, dirs["checkpoints"] / f"checkpoint_{epoch}.pt")
                save_plots(history, dirs["checkpoints"] / "training_metrics.png")
            
            if epoch % args.render_epoch == 0:
                if rms is not None and hasattr(test_env, "obs_rms"):
                    test_env.obs_rms.mean = rms.mean
                    test_env.obs_rms.var = rms.var
                log_video(test_env, agent, device, str(dirs["videos"] / f"epoch_{epoch}.mp4"))

    except KeyboardInterrupt:
        print("\nTraining interrupted.")
    finally:
        save_plots(history, dirs["checkpoints"] / "training_metrics_final.png")
        envs.close()
        test_env.close()
        print("Training completed.")

if __name__ == "__main__":
    train()
