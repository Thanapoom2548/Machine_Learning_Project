import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco

class ArmReacherEnv(gym.Env):
    def __init__(self, xml_path="arm.xml"):
        super().__init__()
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.tip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tip")
        self.target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        
        # Action ของ RL จะเป็นเพียงตัวปรับจูน (Fine-tuning) จากค่า IK
        self.action_space = spaces.Box(low=-0.01, high=0.01, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        
        self.current_step = 0
        self.prev_dist = 0.0

    def _get_obs(self):
        qpos = np.nan_to_num(self.data.qpos[:3].copy(), nan=0.0)
        qvel = np.nan_to_num(self.data.qvel[:3].copy(), nan=0.0)
        tip_pos = np.nan_to_num(self.data.site_xpos[self.tip_site_id].copy(), nan=0.0)
        target_top_pos = np.nan_to_num(self.data.xpos[self.target_body_id].copy(), nan=0.0)
        target_top_pos[2] += 0.02
        
        return np.concatenate([qpos, qvel, tip_pos, target_top_pos]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.6
        self.data.qpos[2] = -0.6
        self.data.ctrl[0] = 0.0
        self.data.ctrl[1] = 0.6
        self.data.ctrl[2] = -0.6
        
        angle = np.random.uniform(-np.pi / 2, np.pi / 2)
        radius = np.random.uniform(0.20, 0.32)
        target_x = radius * np.cos(angle)
        target_y = radius * np.sin(angle)
        target_z = 0.02
        
        self.model.body_pos[self.target_body_id] = [target_x, target_y, target_z]
        mujoco.mj_forward(self.model, self.data)
        
        self.current_step = 0
        tip_pos = self.data.site_xpos[self.tip_site_id]
        target_top = np.array([target_x, target_y, target_z + 0.02])
        self.prev_dist = float(np.linalg.norm(tip_pos - target_top))
        
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        action = np.nan_to_num(action, nan=0.0)
        
        # 1. คำนวณ Inverse Kinematics (IK)
        tip_pos = self.data.site_xpos[self.tip_site_id]
        box_pos = self.data.xpos[self.target_body_id]
        target_top_z = box_pos[2] + 0.02
        target_point = np.array([box_pos[0], box_pos[1], target_top_z])
        
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.tip_site_id)
        
        J = jacp[:, :3] 
        err = target_point - tip_pos
        dq_ik = np.linalg.pinv(J) @ err
        
        # 2. ผสานค่าแกนสมองกล (IK) เข้ากับ RL Action
        ik_step_size = 0.02 
        final_dq = (dq_ik * ik_step_size) + action
        
        self.data.ctrl[0] = np.clip(self.data.ctrl[0] + final_dq[0], -3.14, 3.14)
        self.data.ctrl[1] = np.clip(self.data.ctrl[1] + final_dq[1], -0.2, 1.57)
        self.data.ctrl[2] = np.clip(self.data.ctrl[2] + final_dq[2], -2.2, 0.0)
        
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
            
        if np.isnan(self.data.qpos).any() or np.isnan(self.data.qacc).any():
            mujoco.mj_resetData(self.model, self.data)
            return self._get_obs(), -20.0, True, False, {"d_xy": 1.0, "d_z": 1.0, "status": "exploded"}
            
        tip_pos = self.data.site_xpos[self.tip_site_id]
        d_xy = float(np.linalg.norm(tip_pos[:2] - target_point[:2]))
        d_z = float(tip_pos[2] - target_top_z)
        current_dist = float(np.linalg.norm(tip_pos - target_point))
        
        dist_diff = current_dist - self.prev_dist
        retreat_penalty = 6.0 * dist_diff if dist_diff > 0 else 0.0
        self.prev_dist = current_dist
        
        if self.current_step > 250:
            delay_penalty = 1.5
        elif self.current_step > 150:
            delay_penalty = 0.25
        else:
            delay_penalty = 0.0
        
        # ลงโทษ Action ของ RL อย่างเดียว ไม่ลงโทษฐาน IK
        action_penalty = 0.2 * float(np.sum(np.square(action)))
        vel_penalty = 0.02 * float(np.sum(np.square(self.data.qvel[:3])))
        
        reward = - (2.5 * d_xy + 3.5 * abs(d_z)) - retreat_penalty - delay_penalty - action_penalty - vel_penalty
        
        if tip_pos[2] < 0.02:
            reward -= 2.0
            
        terminated = False
        status = "running"
        
        # ขยายแกน Z เป็น 0.035 และเพิ่มโบนัสเป็น 50 เพื่อเอาชนะความกลัวโดนหักคะแนนชนพื้น
        if d_xy <= 0.035 and 0.0 <= d_z <= 0.035:
            reward += 50.0
            terminated = True
            status = "hit_top_success"
            
        truncated = self.current_step >= 300
        
        return self._get_obs(), reward, terminated, truncated, {"d_xy": d_xy, "d_z": d_z, "status": status}