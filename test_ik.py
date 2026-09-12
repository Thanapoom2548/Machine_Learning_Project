import time
import numpy as np
import mujoco
import mujoco.viewer

L1 = 0.20
L2 = 0.20
Z0 = 0.08

def calculate_ik(x, y, z_world):
    # 1. Base Yaw
    theta_base = np.arctan2(y, x)
    
    # 2. 2D Coordinates
    r = np.sqrt(x**2 + y**2)
    z = z_world - Z0
    d = np.sqrt(r**2 + z**2)
    
    # ป้องกันค่าออกนอกระยะเอื้อมสูงสุด
    d = np.clip(d, 0.05, (L1 + L2) - 1e-4)
    
    # 3. Elbow Pitch
    cos_gamma = (L1**2 + L2**2 - d**2) / (2 * L1 * L2)
    cos_gamma = np.clip(cos_gamma, -1.0, 1.0)
    gamma = np.arccos(cos_gamma)
    theta_elbow = np.pi - gamma  # พับลง
    
    # 4. Shoulder Pitch
    alpha = np.arctan2(-z, r)
    cos_beta = (L1**2 + d**2 - L2**2) / (2 * L1 * d)
    cos_beta = np.clip(cos_beta, -1.0, 1.0)
    beta = np.arccos(cos_beta)
    theta_shoulder = alpha - beta
    
    return theta_base, theta_shoulder, theta_elbow

# โหลดโมเดล
model = mujoco.MjModel.from_xml_path("arm.xml")
data = mujoco.MjData(model)

target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # ดึงพิกัดกล่องแดง
        box_x, box_y, box_z = data.xpos[target_id]
        
        # คำนวณมุมโดยตรงจากสมการ
        q_base, q_shoulder, q_elbow = calculate_ik(box_x, box_y, box_z)
        
        # สั่งกำหนดตำแหน่งตรงๆ (ไม่มีการบวกสะสม)
        data.ctrl[0] = q_base
        data.ctrl[1] = q_shoulder
        data.ctrl[2] = q_elbow
        
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)