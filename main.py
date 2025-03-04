import logging
import time

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter
from pinocchio import pin
from rich.logging import RichHandler

from controllers.cartesian_impedance_controller import CartesianImpedanceController  # noqa: F401
from controllers.gravity_compensation import GravityCompensationController  # noqa: F401
from controllers.operational_space_controller import OperationalSpaceController, OperationalSpaceControllerConfig
from controllers.tsid_controller import TSIDController, TSIDJointConfig, TSIDConfig  # noqa: F401
from controllers.inverse_dynamics_controller import InverseDynamicsController, InverseDynamicsConfig  # noqa: F401

FORMAT = "%(message)s"
logging.basicConfig(level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])

logger = logging.getLogger("rich")


class ThrottleFilter(logging.Filter):
    def __init__(self, name="", interval=5):
        super().__init__(name)
        self.interval = interval
        self.last_log_time = 0

    def filter(self, record):
        now = time.time()
        if now - self.last_log_time >= self.interval:
            self.last_log_time = now
            return True
        return False


logger.addFilter(ThrottleFilter(name="rich", interval=5))

path = "mjcf/scene.xml"
path_urdf = "urdf/fr3_franka_hand.urdf"

model = mujoco.MjModel.from_xml_path(path)
data = mujoco.MjData(model)

# First reset to initialize everything
keyframe_id = 1
mujoco.mj_resetDataKeyframe(model, data, keyframe_id)

dt = model.opt.timestep
max_time = 1000  # [s]

# ctrl = GravityCompensationController(path_urdf)
# ctrl = CartesianImpedanceController(target_frame_name="fr3_hand_tcp", path_to_urdf=path_urdf)
# conf = TSIDJointConfig()
conf = TSIDConfig()
conf.kp_ee = np.array([200.0, 200.0, 200.0, 500.0, 500.0, 2500.0])
conf.kp_posture = 100.0

ctrl = TSIDController(conf=conf, path_to_urdf=path_urdf, dt=dt)  # TODO: use different control and sim frequencies
# conf = InverseDynamicsConfig()
# ctrl = InverseDynamicsController(conf=conf, path_to_urdf=path_urdf)
# conf = OperationalSpaceControllerConfig()
# ctrl = OperationalSpaceController(conf=conf, path_to_urdf=path_urdf)

rate = RateLimiter(frequency=1.0 / dt, warn=False)

logger.info(f"Simulation started with dt - {dt} - and frequency - {1.0 / dt}. Initial qpos: {data.qpos}")

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()

    while viewer.is_running() and time.time() - start < max_time:
        # Set the target pose
        qw, qx, qy, qz = data.mocap_quat[0]
        target_pose = pin.SE3(pin.Quaternion(x=qx, y=qy, z=qz, w=qw), data.mocap_pos[0])
        ctrl.set_target(target_pose, target_q=conf.q0, target_dq=np.zeros(model.nv))

        # Compute the controls of the different controllers
        desired_torques = ctrl.update(data.qpos, data.qvel)

        # Set the control values and simulate a step
        data.ctrl = desired_torques

        mujoco.mj_step(model, data)

        viewer.sync()

        logger.info(f"Robot state: {data.qpos}, Mocap Pos: {data.mocap_pos[0]}")

        rate.sleep()
