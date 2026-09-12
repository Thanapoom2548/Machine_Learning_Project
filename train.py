import time
import torch
import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from reacher_env import ArmReacherEnv

def make_env(xml_path="arm.xml"):
    def _init():
        return ArmReacherEnv(xml_path=xml_path)
    return _init

if __name__ == "__main__":
    # 1. ตรวจสอบการใช้งาน CUDA (RTX 3060 Ti)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== กำลังฝึกโมเดลบน: {device.upper()} ===")
    if device == "cuda":
        print(f"GPU Detected: {torch.cuda.get_device_name(0)}")

    # 2. ปรับตามสเปค i3-12100F (4 Core / 8 Threads)
    # รัน 7 Environments แบบคู่ขนาน พร้อมกันบน 7 Threads
    num_cpu = 7
    print(f"กำลังสร้าง {num_cpu} Parallel Environments บน CPU Multi-Threading...")
    env_fns = [make_env("arm.xml") for _ in range(num_cpu)]
    vec_env = SubprocVecEnv(env_fns)

    # 3. กำหนดค่า PPO ให้เหมาะกับการประมวลผลบน GPU
    # n_steps=2048 x 7 envs = 14,336 samples ต่อรอบ
    # batch_size=256 ส่งเข้า Tensor Cores ของ RTX 3060 Ti
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        device=device,
        verbose=1
    )

    # 4. ฝึกจำนวน 1,000,000 timesteps (ใช้เวลาประมาณ 1-2 นาทีเท่านั้น)
    print("--- เริ่มกระบวนการฝึก (Training Phase) ---")
    start_time = time.time()
    model.learn(total_timesteps=1000000)
    train_duration = time.time() - start_time
    print(f"--- การฝึกเสร็จสิ้น ใช้เวลาไปทั้งสิ้น {train_duration:.2f} วินาที ---")

    # บันทึกโมเดล
    model.save("ppo_reacher_optimized")
    vec_env.close()

    # 5. เปิดดูผลลัพธ์บน MuJoCo Viewer
    print("กำลังเปิดการจำลองเพื่อดูผลลัพธ์...")
    eval_env = ArmReacherEnv(xml_path="arm.xml")
    obs, _ = eval_env.reset()

    with mujoco.viewer.launch_passive(eval_env.model, eval_env.data) as viewer:
        while viewer.is_running():
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)

            viewer.sync()
            time.sleep(0.02)

            if terminated or truncated:
                print(f"จบรอบ! ระยะห่างปลายแขนกับกล่อง: {info['dist'] * 100:.2f} ซม.")
                obs, _ = eval_env.reset()