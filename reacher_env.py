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
        
        self.action_space = spaces.Box(low=-0.04, high=0.04, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        self.current_step = 0

    def _get_obs(self):
        # ดักค่า NaN ไม่ให้หลุดเข้าไปทำลาย Weights ของ Neural Network
        qpos = np.nan_to_num(self.data.qpos[:3].copy(), nan=0.0)
        tip_pos = np.nan_to_num(self.data.site_xpos[self.tip_site_id].copy(), nan=0.0)
        target_top_pos = np.nan_to_num(self.data.xpos[self.target_body_id].copy(), nan=0.0)
        target_top_pos[2] += 0.02
        return np.concatenate([qpos, tip_pos, target_top_pos]).astype(np.float32)

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
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        
        # ป้องกัน Action ขยะ
        action = np.nan_to_num(action, nan=0.0)
        
        self.data.ctrl[0] = np.clip(self.data.ctrl[0] + action[0], -3.14, 3.14)
        self.data.ctrl[1] = np.clip(self.data.ctrl[1] + action[1], -0.2, 1.57)
        self.data.ctrl[2] = np.clip(self.data.ctrl[2] + action[2], -2.2, 0.0)
        
        for _ in range(3):
            mujoco.mj_step(self.model, self.data)
            
        # 1. ถ้าฟิสิกส์ระเบิด (QACC หลุด) ให้โยนค่า Penalty และเริ่มใหม่ทันที
        if np.isnan(self.data.qpos).any() or np.isnan(self.data.qacc).any():
            mujoco.mj_resetData(self.model, self.data)
            return self._get_obs(), -20.0, True, False, {"d_xy": 1.0, "d_z": 1.0, "status": "exploded"}
            
        # 2. ถ้าระบบปกติ ให้คำนวณระยะทางและส่งค่า Return ตัวจริง (ส่วนที่แหว่งไปรอบก่อน)
        tip_pos = self.data.site_xpos[self.tip_site_id]
        box_pos = self.data.xpos[self.target_body_id]
        target_top_z = box_pos[2] + 0.02
        
        d_xy = np.linalg.norm(tip_pos[:2] - box_pos[:2])
        d_z = tip_pos[2] - target_top_z
        
        reward = - (d_xy + abs(d_z))
        
        # หักคะแนนถ้าแขนทิ่มลงพื้น
        if tip_pos[2] < 0.02:
            reward -= 2.0
            
        terminated = False
        status = "running"
        
        # เงื่อนไขแตะฝาบนสำเร็จ
        if d_xy <= 0.03 and 0.0 <= d_z <= 0.025:
            reward += 50.0
            terminated = True
            status = "hit_top_success"
            
        truncated = self.current_step >= 150
        obs = self._get_obs()
        
        return obs, reward, terminated, truncated, {"d_xy": d_xy, "d_z": d_z, "status": status}