import threading
import time

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter
from pinocchio import pin

from controllers.controller import Controller, ControllerConfig
from controllers.util import make_controller, make_throttled_logger


logger = make_throttled_logger("main", interval=5)

path = "mjcf/scene.xml"
path_urdf = "urdf/fr3_franka_hand.urdf"
ctrl_type = "gravity_compensation"
multi_threading = False
sim_dt = 0.002
max_time = 1000  # [s]

spec = mujoco.MjSpec.from_file(path)
spec.option.timestep = sim_dt
model = spec.compile()
data = mujoco.MjData(model)

# First reset to initialize everything
keyframe_id = 1
mujoco.mj_resetDataKeyframe(model, data, keyframe_id)


ctrl, conf = make_controller(ctrl_type, path_urdf, sim_dt)
rate = RateLimiter(frequency=1.0 / sim_dt, warn=False)

# logger.info(f"Simulation started with dt - {sim_dt} - and frequency - {1.0 / sim_dt}. Initial qpos: {data.qpos}")


def control_step(ctrl: Controller, conf: ControllerConfig, model: mujoco.MjModel, data: mujoco.MjData):
    # Set the target pose
    qw, qx, qy, qz = data.mocap_quat[0]
    target_pose = pin.SE3(pin.Quaternion(x=qx, y=qy, z=qz, w=qw), data.mocap_pos[0])
    ctrl.set_target(target_pose, target_q=conf.q0, target_dq=np.zeros(model.nv))

    # Compute the controls of the different controllers
    desired_torques = ctrl.update(data.time, data.qpos, data.qvel)

    # Set the control values and simulate a step
    data.ctrl = desired_torques


def controller_loop(viewer, ctrl, conf, model, data):
    last_ctrl_time = 0

    while viewer.is_running():
        if data.time > last_ctrl_time + conf.ctrl_dt:
            last_ctrl_time = data.time
            control_step(ctrl, conf, model, data)

        time.sleep(data.time - last_ctrl_time)


with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()

    if multi_threading:
        ctrl_thread = threading.Thread(target=controller_loop, args=(viewer, ctrl, conf, model, data))
        ctrl_thread.start()

    while viewer.is_running() and time.time() - start < max_time:
        mujoco.mj_step(model, data)

        if not multi_threading:
            control_step(ctrl, conf, model, data)

        viewer.sync()

        # logger.info(f"Robot state: {data.qpos}, Mocap Pos: {data.mocap_pos[0]}")
        logger.info(f"Current sim-time : {data.time} - Current time from start: {time.time() - start}")

        rate.sleep()

ctrl_thread.join()
