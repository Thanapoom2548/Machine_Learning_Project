import time
import mujoco
import mujoco.viewer
from stable_baselines3 import SAC
from reacher_env import ArmReacherEnv

def main():
    print("กำลังเตรียมสภาพแวดล้อมและโหลดโมเดล...")
    # 1. โหลด Environment และ โมเดลที่เทรนได้ 100%
    env = ArmReacherEnv("arm.xml")
    model = SAC.load("sac_residual_model.zip")
    
    print("กำลังเปิดหน้าต่างจำลองภาพ 3 มิติ... (กด ESC หรือปิดหน้าต่างเพื่อออก)")

    # 2. เปิด Viewer ของ MuJoCo เพื่อดูภาพ
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        # ลองรันให้ดู 10 รอบ (Episodes)
        for ep in range(10):  
            obs, _ = env.reset()
            done = False
            steps = 0
            
            print(f"--- เริ่มรอบที่ {ep+1} ---")
            
            # รันไปเรื่อยๆ จนกว่าจะแตะสำเร็จ หรือหน้าต่างถูกปิด
            while not done and viewer.is_running():
                # ให้ AI ตัดสินใจจาก State ปัจจุบัน (deterministic=True เพื่อดึงความแม่นยำสูงสุดออกมา)
                action, _ = model.predict(obs, deterministic=True)
                
                # ส่ง Action ไปขยับแขนกล
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1
                
                # ซิงก์ภาพกราฟิกให้ตรงกับสถานะคำนวณ
                viewer.sync()
                
                # หน่วงเวลา 0.02 วินาที เพื่อให้ตาคนมองทัน (หากไม่หน่วง แขนจะวาร์ปไปแตะจบในเสี้ยววินาที)
                time.sleep(0.02)
                
            if terminated:
                print(f"✅ แตะสำเร็จใน {steps} สเต็ป!")
            elif truncated:
                print(f"❌ หมดเวลา (ลอยค้าง)")
            
            # หยุดพักภาพ 1 วินาทีก่อนสุ่มตำแหน่งกล่องเพื่อเริ่มรอบใหม่
            time.sleep(1.0)

if __name__ == "__main__":
    main()