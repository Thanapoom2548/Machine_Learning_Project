import os
import time
from datetime import datetime
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from reacher_env import ArmReacherEnv

def make_env(xml_path="arm.xml"):
    def _init():
        env = ArmReacherEnv(xml_path=xml_path)
        return Monitor(env)  # หุ้ม Env ด้วย Monitor เพื่อเก็บคะแนนและสถิติ
    return _init

if __name__ == '__main__':
    device = "cpu"
    print(f"=== กำลังฝึกโมเดลบน: {device.upper()} (บังคับปิด GPU เพื่อเพิ่ม FPS) ===")
    
    num_envs = 8
    vec_env = SubprocVecEnv([make_env("arm.xml") for _ in range(num_envs)])
    
    model = SAC(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        buffer_size=100000,
        batch_size=256,
        train_freq=(512, "step"),
        gamma=0.99,
        tau=0.005,
        device=device,
        verbose=1,
        tensorboard_log="./tensorboard_logs/"
    )
    
    TOTAL_TIMESTEPS = 500000 
    print(f"--- เริ่มการฝึกสอนโมเดล ({TOTAL_TIMESTEPS:,} Timesteps) ---")
    start_time = time.time()
    
    # หน่วงการพิมพ์ลงหน้าจอให้เหลือทุกๆ 50 Episodes
    model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=50)
    
    train_duration = time.time() - start_time
    print(f"--- การฝึกเสร็จสิ้น ใช้เวลาไปทั้งสิ้น {train_duration:.2f} วินาที ---")
    
    model_save_path = "sac_residual_model.zip"
    model.save(model_save_path)
    print(f"บันทึกโมเดลไว้ที่: {model_save_path}")
    vec_env.close()
    
    print("\n--- เริ่มขั้นตอนการประเมินผล (Evaluation Phase 100 Episodes) ---")
    eval_env = ArmReacherEnv("arm.xml")
    
    eval_episodes = 100
    success_count = 0
    side_hit_count = 0
    timeout_count = 0
    
    xy_errors = []
    z_errors = []
    dist_3d_errors = []
    step_counts = []
    episode_returns = []
    
    for ep in range(eval_episodes):
        obs, _ = eval_env.reset()
        done = False
        ep_ret = 0.0
        step_cnt = 0
        final_info = {}
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_ret += reward
            step_cnt += 1
            done = terminated or truncated
            final_info = info
            
        episode_returns.append(ep_ret)
        step_counts.append(step_cnt)
        
        d_xy = final_info.get("d_xy", 1.0)
        d_z = final_info.get("d_z", 1.0)
        d_3d = np.sqrt(d_xy**2 + d_z**2)
        status = final_info.get("status", "running")
        
        xy_errors.append(d_xy)
        z_errors.append(abs(d_z))
        dist_3d_errors.append(d_3d)
        
        if status == "hit_top_success":
            success_count += 1
        elif status == "side_collision":
            side_hit_count += 1
        else:
            timeout_count += 1

    eval_env.close()

    success_rate = (success_count / eval_episodes) * 100
    side_hit_rate = (side_hit_count / eval_episodes) * 100
    timeout_rate = (timeout_count / eval_episodes) * 100
    
    report_content = f"""======================================================================
               รายงานผลการประเมินแบบผสมผสาน IK + SAC
======================================================================
วันที่และเวลาบันทึก: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
อัลกอริทึม: Soft Actor-Critic (SAC) + Inverse Kinematics
สภาพแวดล้อม: Custom 3-DoF Reacher (MuJoCo Physics Engine)
ฮาร์ดแวร์ประมวลผล: CPU | {num_envs} Workers
เวลาที่ใช้ฝึกทั้งหมด: {train_duration:.2f} วินาที ({TOTAL_TIMESTEPS:,} Timesteps)
จำนวนรอบการทดสอบประเมิน: {eval_episodes} Episodes

----------------------------------------------------------------------
1. ตัวชี้วัดอัตราความสำเร็จ (Success Rate Metrics)
----------------------------------------------------------------------
- อัตราการแตะฝาบนสำเร็จ (Success Rate):      {success_rate:.2f} % ({success_count}/{eval_episodes})
- อัตราการเกิดข้อผิดพลาดชนข้างกล่อง (Side Hits): {side_hit_rate:.2f} % ({side_hit_count}/{eval_episodes})
- อัตราการหมดเวลาเอื้อมไม่ถึง (Timeouts):       {timeout_rate:.2f} % ({timeout_count}/{eval_episodes})

----------------------------------------------------------------------
2. ตัวชี้วัดความแม่นยำเชิงตำแหน่ง (Positional Error Metrics)
----------------------------------------------------------------------
- ความคลาดเคลื่อนแนวราบเฉลี่ย (Mean XY Error): {np.mean(xy_errors)*100:.2f} cm
- ความคลาดเคลื่อนแนวดิ่งเฉลี่ย (Mean Z Error):  {np.mean(z_errors)*100:.2f} cm
- ความคลาดเคลื่อนระยะขจัดรวม (Mean 3D Error): {np.mean(dist_3d_errors)*100:.2f} cm
======================================================================
"""
    print(report_content)
    with open("evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write(report_content)