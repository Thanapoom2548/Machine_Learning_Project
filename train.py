import os
import time
from datetime import datetime
import torch
import numpy as np
import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from reacher_env import ArmReacherEnv

def make_env(xml_path="arm.xml"):
    def _init():
        return ArmReacherEnv(xml_path=xml_path)
    return _init

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== กำลังฝึกโมเดลบน: {device.upper()} ===")

    # รัน 7 Process คู่ขนานบน i3-12100F
    num_cpu = 8
    env_fns = [make_env("arm.xml") for _ in range(num_cpu)]
    vec_env = SubprocVecEnv(env_fns)

    # คอนฟิก PPO สำหรับ RTX 3060 Ti
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=512,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.001,
        device=device,
        verbose=1
    )

    # ฝึกสอนโมเดล 500,000 timesteps
    print("--- เริ่มการฝึกสอนโมเดล (Training Phase) ---")
    start_time = time.time()
    model.learn(total_timesteps=1000000)
    train_duration = time.time() - start_time
    print(f"--- การฝึกเสร็จสิ้น ใช้เวลาไปทั้งสิ้น {train_duration:.2f} วินาที ---")

    model.save("ppo_top_touch_model")
    vec_env.close()

    # --- ส่วนการประเมินผลเชิงตัวเลข (Evaluation Phase) ---
    print("\n--- กำลังทดสอบโมเดล 100 รอบ เพื่อเก็บข้อมูลทางสถิติ ---")
    eval_env = ArmReacherEnv(xml_path="arm.xml")
    total_episodes = 100

    success_count = 0
    side_hit_count = 0
    timeout_count = 0
    
    list_steps = []
    list_rewards = []
    list_xy_errors = []
    list_z_errors = []

    for ep in range(total_episodes):
        obs, _ = eval_env.reset()
        done = False
        ep_reward = 0.0
        step_count = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_reward += reward
            step_count += 1
            done = terminated or truncated

            if done:
                list_steps.append(step_count)
                list_rewards.append(ep_reward)
                list_xy_errors.append(info["d_xy"] * 100)  # แปลงเป็น cm
                list_z_errors.append(abs(info["d_z"]) * 100)  # แปลงเป็น cm

                if info["status"] == "hit_top_success":
                    success_count += 1
                elif info["status"] == "hit_side":
                    side_hit_count += 1
                else:
                    timeout_count += 1

    # คำนวณค่าเฉลี่ยทางสถิติ
    success_rate = (success_count / total_episodes) * 100
    side_hit_rate = (side_hit_count / total_episodes) * 100
    timeout_rate = (timeout_count / total_episodes) * 100
    
    avg_steps = float(np.mean(list_steps))
    avg_reward = float(np.mean(list_rewards))
    avg_xy_err = float(np.mean(list_xy_errors))
    avg_z_err = float(np.mean(list_z_errors))
    total_euclidean_err = float(np.mean(np.sqrt(np.array(list_xy_errors)**2 + np.array(list_z_errors)**2)))

    # --- สร้างรายงานเป็นไฟล์ TXT ---
    report_filename = "evaluation_report.txt"
    report_content = f"""======================================================================
               รายงานผลการประเมินโมเดลควบคุมแขนกล 3-DOF (RL-PPO)
======================================================================
วันที่และเวลาบันทึก: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
อัลกอริทึม: Proximal Policy Optimization (PPO)
สภาพแวดล้อม: Custom 3-DoF Reacher (MuJoCo Physics Engine)
เงื่อนไขงาน: แตะเฉพาะฝาบนกล่องเป้าหมาย (Top-Surface Touch Only)
ฮาร์ดแวร์ประมวลผล: i3-12100F (7 Subproc Workers) + RTX 3060 Ti ({device.upper()})
เวลาที่ใช้ฝึกทั้งหมด: {train_duration:.2f} วินาที (500,000 Timesteps)
จำนวนรอบการทดสอบประเมิน: {total_episodes} Episodes

----------------------------------------------------------------------
1. ตัวชี้วัดอัตราความสำเร็จ (Success Rate Metrics)
----------------------------------------------------------------------
- อัตราการแตะฝาบนสำเร็จ (Success Rate):      {success_rate:.2f} % ({success_count}/{total_episodes})
- อัตราการเกิดข้อผิดพลาดชนข้างกล่อง (Side Hits): {side_hit_rate:.2f} % ({side_hit_count}/{total_episodes})
- อัตราการหมดเวลาเอื้อมไม่ถึง (Timeouts):       {timeout_rate:.2f} % ({timeout_count}/{total_episodes})

----------------------------------------------------------------------
2. ตัวชี้วัดความแม่นยำเชิงตำแหน่ง (Positional Error Metrics)
----------------------------------------------------------------------
- ความคลาดเคลื่อนแนวราบเฉลี่ย (Mean XY Error): {avg_xy_err:.2f} cm
- ความคลาดเคลื่อนแนวดิ่งเฉลี่ย (Mean Z Error):  {avg_z_err:.2f} cm
- ความคลาดเคลื่อนระยะขจัดรวม (Mean 3D Error): {total_euclidean_err:.2f} cm

----------------------------------------------------------------------
3. ตัวชี้วัดประสิทธิภาพการเคลื่อนที่ (Kinematic & Return Metrics)
----------------------------------------------------------------------
- จำนวนสเต็ปเฉลี่ยในการแตะสำเร็จ (Avg Steps):    {avg_steps:.2f} Steps/Episode
- ผลตอบแทนเฉลี่ยต่อรอบ (Mean Episodic Return):  {avg_reward:.2f}
======================================================================
"""

    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\n[สำเร็จ] บันทึกผลการประเมินลงไฟล์ '{report_filename}' เรียบร้อยแล้ว")
    print(report_content)

    # --- เปิด Viewer แสดงผล 3 มิติ ---
    print("กำลังเปิดการจำลอง 3 มิติเพื่อดูพฤติกรรมจริง...")
    obs, _ = eval_env.reset()
    with mujoco.viewer.launch_passive(eval_env.model, eval_env.data) as viewer:
        while viewer.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            viewer.sync()
            time.sleep(0.02)
            if terminated or truncated:
                obs, _ = eval_env.reset()