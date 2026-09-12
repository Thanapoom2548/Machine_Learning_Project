import time
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("arm.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    print("โหลดโมเดลสำเร็จ กำลังเปิด Viewer...")
    while viewer.is_running():
        # ทดสอบส่งคำสั่งมุมเริ่มต้น
        data.ctrl[0] = 0.0   # Base
        data.ctrl[1] = 0.3   # Shoulder (ก้มลงนิดหน่อย)
        data.ctrl[2] = -0.6  # Elbow (งอศอกลง)

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)