import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco

class ArmReacherEnv(gym.Env):
    def __init__(self, xml_path="arm.xml"):
        super().__init__()
        
        # 1. โหลดโมเดล MuJoCo
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.tip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tip")
        self.target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        
        # 2. กำหนด Action Space: ปรับมุม 3 มอเตอร์ (เอว, ไหล่, ศอก) ค่าอยู่ในช่วง [-0.05, 0.05] rad ต่อ step
        self.action_space = spaces.Box(low=-0.05, high=0.05, shape=(3,), dtype=np.float32)
        
        # 3. กำหนด Observation Space: 9 มิติ
        # [มุม 3 ข้อต่อ (qpos), พิกัดปลายเขียว (tip: x,y,z), พิกัดกล่องแดง (target: x,y,z)]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)

    def _get_obs(self):
        qpos = self.data.qpos[:3].copy()
        tip_pos = self.data.site_xpos[self.tip_site_id].copy()
        target_pos = self.data.xpos[self.target_body_id].copy()
        return np.concatenate([qpos, tip_pos, target_pos]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # รีเซ็ตมุมเริ่มต้นของแขน
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = -0.5
        
        # สุ่มตำแหน่งกล่องแดงใหม่ในระยะเอื้อมที่ปลอดภัย
        # รัศมีแนวราบ r อยู่ระหว่าง 0.15 ถึง 0.35 เมตร, สูง z อยู่ระหว่าง 0.02 ถึง 0.15 เมตร
        angle = np.random.uniform(-np.pi / 2, np.pi / 2)
        radius = np.random.uniform(0.18, 0.32)
        target_x = radius * np.cos(angle)
        target_y = radius * np.sin(angle)
        target_z = np.random.uniform(0.02, 0.15)
        
        self.model.body_pos[self.target_body_id] = [target_x, target_y, target_z]
        
        mujoco.mj_forward(self.model, self.data)
        self.current_step = 0
        
        return self._get_obs(), {}

    def step(self, action):
        # 1. พิกัดกึ่งกลางฝาบนของกล่อง
target_top = target_pos.copy()
target_top[2] += 0.02  # ครึ่งหนึ่งของความสูงกล่อง

# 2. แยกคิดระยะแนวนอน (d_xy) และระยะแนวดิ่ง (d_z)
d_xy = np.linalg.norm(tip_pos[:2] - target_top[:2])
d_z = tip_pos[2] - target_top[2]

# 3. คำนวณ Reward
reward = -(d_xy + np.abs(d_z))

# ตรวจจับการชนข้างกล่อง: ปลายแขนต่ำกว่าฝาบน แต่ยังไม่อยู่ในพื้นที่กึ่งกลาง
terminated = False
if d_z < 0 and d_xy > 0.02:
    reward -= 5.0  # โดนหักคะแนนจากการชนข้างกล่อง
    terminated = True

# ตรวจจับการแตะสำเร็จบนฝาบน: แนวราบตรง และแตะลงมาจากด้านบนพอดี
elif d_xy < 0.02 and 0 <= d_z <= 0.015:
    reward += 15.0  # โบนัสแตะฝาบนสำเร็จ
    terminated = True
        
        return obs, reward, terminated, truncated, {"dist": dist}