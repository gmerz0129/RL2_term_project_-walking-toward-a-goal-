from dataclasses import asdict, dataclass
from typing import Tuple
import gymnasium as gym
import numpy as np
import mujoco

@dataclass
class RewardTerms:
    # 보상 구성 요소를 저장하는 data class.
    base: float = 0.0
    progress: float = 0.0
    velocity: float = 0.0
    heading: float = 0.0
    upright: float = 0.0
    distance_reward: float = 0.0
    medium_bonus: float = 0.0
    
    # Costs
    hole_penalty: float = 0.0
    action_cost: float = 0.0
    joint_vel_cost: float = 0.0
    contact_cost: float = 0.0
    lateral_cost: float = 0.0
    smooth_cost: float = 0.0
    bonus: float = 0.0


class StandingResetWrapper(gym.Wrapper):
    # 휴머노이드를 안정적인 서있는 자세로 초기화하는 래퍼.
    
    def __init__(
        self,
        env: gym.Env,
        pose_noise: float = 1e-4,
        vel_noise: float = 1e-5,
        settle_steps: int = 4,
    ) -> None:
        super().__init__(env)
        self.pose_noise = pose_noise
        self.vel_noise = vel_noise
        self.settle_steps = settle_steps
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        sim = self.env.unwrapped
        
        init_qpos = sim.init_qpos.copy()
        init_qvel = sim.init_qvel.copy()
        
        if self.pose_noise > 0:
            init_qpos += self.np_random.uniform(-self.pose_noise, self.pose_noise, size=init_qpos.shape)
        if self.vel_noise > 0:
            init_qvel += self.np_random.uniform(-self.vel_noise, self.vel_noise, size=init_qvel.shape)
        
        sim.set_state(init_qpos, init_qvel)
        
        for _ in range(self.settle_steps):
            sim.do_simulation(np.zeros(sim.model.nu), sim.frame_skip)
        
        return sim._get_obs(), info


class GoalRewardWrapper(gym.Wrapper):

    def __init__(
        self,
        env: gym.Env,
        # Reward Flags
        use_base_reward: bool = False,
        use_progress_reward: bool = False,
        use_velocity_reward: bool = False,
        use_heading_reward: bool = False,
        use_upright_reward: bool = False,
        use_distance_reward: bool = False,
        use_medium_bonus_reward: bool = False,
        
        # Cost Flags
        use_hole_penalty: bool = False,
        use_action_penalty: bool = False,
        use_joint_vel_penalty: bool = False,
        use_contact_penalty: bool = False,
        use_lateral_penalty: bool = False,
        use_smooth_penalty: bool = False,
        
        # Parameters
        goal_xy: Tuple[float, float] = (7.5, 2.5),
        start_xy: Tuple[float, float] = (-1.5, -2.0),
        goal_radius: float = 0.75,
        
        progress_weight: float = 50.0,
        velocity_weight: float = 10.0,
        heading_weight: float = 3.0,
        upright_weight: float = 2.0,
        distance_reward_scale: float = 5.0,
        
        hole_penalty_weight: float = 50.0,
        action_penalty: float = 0.002,
        joint_velocity_penalty: float = 0.0001,
        contact_penalty: float = 5e-5,
        contact_threshold: float = 1200.0,
        lateral_penalty: float = 0.1,
        smooth_penalty: float = 0.0005,
        
        goal_bonus: float = 1000.0,
        medium_bonus: float = 50.0,
        
        yaw_fail_threshold: float = np.deg2rad(150),
        yaw_fail_penalty: float = 10.0,
        fail_height: float = 0.6,
        early_stop_penalty: float = 50.0,
        min_progress_per_step: float = 0.08,
        reward_scale: float = 1.0,
    ) -> None:
        super().__init__(env)
        
        # Store Flags
        self.use_base_reward = use_base_reward
        self.use_progress_reward = use_progress_reward
        self.use_velocity_reward = use_velocity_reward
        self.use_heading_reward = use_heading_reward
        self.use_upright_reward = use_upright_reward
        self.use_distance_reward = use_distance_reward
        self.use_medium_bonus_reward = use_medium_bonus_reward
        
        self.use_hole_penalty = use_hole_penalty
        self.use_action_penalty = use_action_penalty
        self.use_joint_vel_penalty = use_joint_vel_penalty
        self.use_contact_penalty = use_contact_penalty
        self.use_lateral_penalty = use_lateral_penalty
        self.use_smooth_penalty = use_smooth_penalty

        self.goal_xy = np.array(goal_xy, dtype=np.float64)
        self.start_xy = np.array(start_xy, dtype=np.float64)
        self.goal_radius = goal_radius
        
        # 파라미터 저장
        self.params = {k: v for k, v in locals().items() if k not in ['self', 'env', '__class__'] and not k.startswith('use_')}
        
        # 구멍 + 벽 Geom ID 찾기
        self.hazard_geom_names = [
            "hole_marker_1", "hole_marker_2", 
            "wall_left", "wall_right", "wall_top", "wall_bottom"
        ]
        self.hazard_geom_ids = []

        self._find_hazard_geoms() # 초기화 시 시도

        # 경로 계산
        self.path_length = float(np.linalg.norm(self.goal_xy - self.start_xy))
        self.min_travel_steps = max(50, int(np.ceil(self.path_length / min_progress_per_step)))
        self.heading_fail_grace_steps = max(1, int(0.3 * self.min_travel_steps))
        
        self.medium_checkpoints = [0.25, 0.5, 0.75]
        self.medium_bonus_reached = [False] * 3
        
        self.prev_distance = None
        self.prev_action = None
        self.step_count = 0
    
    def _find_hazard_geoms(self):
        self.hazard_geom_ids = []
        try:
            sim = self.env.unwrapped
            model = sim.model
            
            for name in self.hazard_geom_names:
                found_id = -1
                # 최신 Mujoco 바인딩
                if hasattr(mujoco, 'mj_name2id'):
                    found_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                
                # Mujoco-py 바인딩
                elif hasattr(model, 'geom_name2id'):
                    found_id = model.geom_name2id(name)
                
                if found_id != -1:
                    self.hazard_geom_ids.append(found_id)
                else:
                    print(f"Geom '{name}' not found in model.")
                    
        except Exception as e:
            print(f"Error finding hazard geoms: {e}")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        pos = self._get_pos(info)
        self.prev_distance = float(np.linalg.norm(self.goal_xy - pos))
        self.prev_action = None
        self.step_count = 0
        self.medium_bonus_reached = [False] * 3
        
        info.update({
            "distance_to_goal": self.prev_distance,
            "is_success": False,
            "min_travel_steps": self.min_travel_steps,
            "medium_bonus_reached": 0
        })
        return obs, info
    
    def step(self, action):
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        sim = self.env.unwrapped
        
        pos = self._get_pos(info)
        to_goal = self.goal_xy - pos
        dist = float(np.linalg.norm(to_goal))
        direction = to_goal / (dist + 1e-9)
        
        self.step_count += 1
        prev_dist = self.prev_distance if self.prev_distance is not None else dist
        progress = prev_dist - dist
        
        p = self.params
        
        # 보상 계산
        rewards = {
            "base": base_reward if self.use_base_reward else 0.0,
            "progress": (p['progress_weight'] * progress) if self.use_progress_reward else 0.0,
            "velocity": (p['velocity_weight'] * max(0.0, float(np.dot([info.get("x_velocity", 0), info.get("y_velocity", 0)], direction)))) if self.use_velocity_reward else 0.0,
            "heading": self._calc_heading_reward(to_goal) if self.use_heading_reward else 0.0,
            "upright": self._calc_upright_reward(sim) if self.use_upright_reward else 0.0,
            "distance": (p['distance_reward_scale'] * max(0.0, 1.0 - (dist / self.path_length))) if self.use_distance_reward else 0.0,
            "medium_bonus": self._check_medium_bonus(dist) if self.use_medium_bonus_reward else 0.0
        }
        
        # 페널티 계산
        action_arr = np.asarray(action, dtype=np.float64)
        
        # 구멍 + 벽접촉 체크
        is_hazard_contact = self._check_hazard_contact(sim)
        
        # 페널티 점수는 플래그가 켜져 있을 때만 부여
        hazard_penalty = p['hole_penalty_weight'] if (is_hazard_contact and self.use_hole_penalty) else 0.0
        
        penalties = {
            "hole": hazard_penalty, 
            "action": (p['action_penalty'] * float(np.sum(np.square(action_arr)))) if self.use_action_penalty else 0.0,
            "joint_vel": (p['joint_velocity_penalty'] * float(np.mean(np.square(sim.data.qvel)))) if self.use_joint_vel_penalty else 0.0,
            "contact": self._calc_contact_cost(sim) if self.use_contact_penalty else 0.0,
            "lateral": (p['lateral_penalty'] * self._lateral_deviation(pos)) if self.use_lateral_penalty else 0.0,
            "smooth": (p['smooth_penalty'] * (float(np.sum(np.square(action_arr - self.prev_action))) if self.prev_action is not None else 0.0)) if self.use_smooth_penalty else 0.0
        }
        
        shaped_reward = sum(rewards.values()) - sum(penalties.values())
        
        # 종료 조건
        if is_hazard_contact:
            terminated = True
        
        success = dist < self.goal_radius
        bonus = 0.0
        if success and self.step_count >= self.min_travel_steps:
            bonus = p['goal_bonus']
            shaped_reward += bonus
            terminated = True
        elif success:
            shaped_reward -= p['early_stop_penalty']
            terminated = True
        
        if sim.data.qpos[2] < p['fail_height']:
            terminated = True
            shaped_reward -= 50.0
            
        heading_diff = self._get_heading_diff(to_goal)
        if abs(heading_diff) > p['yaw_fail_threshold'] and self.step_count >= self.heading_fail_grace_steps:
            terminated = True
            shaped_reward -= p['yaw_fail_penalty']
            
        shaped_reward *= p['reward_scale']
        
        # Info
        reward_terms = RewardTerms(
            base=rewards['base'],
            progress=rewards['progress'], velocity=rewards['velocity'], heading=rewards['heading'],
            upright=rewards['upright'], distance_reward=rewards['distance'], 
            medium_bonus=rewards['medium_bonus'],
            hole_penalty=penalties['hole'],
            action_cost=penalties['action'], joint_vel_cost=penalties['joint_vel'],
            contact_cost=penalties['contact'], lateral_cost=penalties['lateral'], smooth_cost=penalties['smooth'],
            bonus=bonus
        )
        
        info.update({
            "distance_to_goal": dist,
            "reward_terms": asdict(reward_terms),
            "is_success": success and self.step_count >= self.min_travel_steps,
            "medium_bonus_reached": sum(self.medium_bonus_reached),
            "is_hole_fall": is_hazard_contact
        })
        
        self.prev_distance = dist
        self.prev_action = action_arr.copy()
        
        return obs, shaped_reward, terminated, truncated, info

    # 이외 매소드
    def _get_pos(self, info):
        return np.array([info.get("x_position", 0.0), info.get("y_position", 0.0)], dtype=np.float64)
    
    # 구멍/벽 충돌 감지
    def _check_hazard_contact(self, sim):
        if not self.hazard_geom_ids:
            return False
            
        for i in range(sim.data.ncon):
            contact = sim.data.contact[i]
            if contact.geom1 in self.hazard_geom_ids or contact.geom2 in self.hazard_geom_ids:
                return True
        return False

    def _calc_heading_reward(self, to_goal):
        yaw = self._heading_yaw()
        target = float(np.arctan2(to_goal[1], to_goal[0]))
        diff = np.arctan2(np.sin(target - yaw), np.cos(target - yaw))
        return self.params['heading_weight'] * ((np.cos(diff) + 1.0) * 0.5)

    def _get_heading_diff(self, to_goal):
        yaw = self._heading_yaw()
        target = float(np.arctan2(to_goal[1], to_goal[0]))
        return np.arctan2(np.sin(target - yaw), np.cos(target - yaw))

    def _heading_yaw(self):
        q = self.env.unwrapped.data.qpos[3:7]
        return float(np.arctan2(2*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]**2 + q[3]**2)))

    def _calc_upright_reward(self, sim):
        z = float(sim.data.qpos[2])
        min_z, max_z = getattr(sim, '_healthy_z_range', (0.9, 2.0))
        return self.params['upright_weight'] * np.clip((z - min_z) / (max_z - min_z), 0.0, 1.0)

    def _check_medium_bonus(self, dist):
        reward = 0.0
        traveled = self.path_length - dist
        for i, m in enumerate(self.medium_checkpoints):
            if not self.medium_bonus_reached[i] and traveled >= m * self.path_length:
                reward += self.params['medium_bonus']
                self.medium_bonus_reached[i] = True
        return reward

    def _calc_contact_cost(self, sim):
        forces = np.linalg.norm(sim.data.cfrc_ext[1:], axis=1)
        return self.params['contact_penalty'] * float(np.sum(np.maximum(0, forces - self.params['contact_threshold'])))

    def _lateral_deviation(self, pos):
        path_vec = self.goal_xy - self.start_xy
        path_len = np.linalg.norm(path_vec)
        if path_len < 1e-6: return 0.0
        proj = np.clip(np.dot(pos - self.start_xy, path_vec / path_len), 0.0, path_len)
        closest = self.start_xy + proj * (path_vec / path_len)
        return float(np.linalg.norm(pos - closest))
