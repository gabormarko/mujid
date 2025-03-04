import tsid
import numpy as np
import pinocchio as pin
from dataclasses import dataclass
from controllers.controller import Controller


@dataclass
class InverseDynamicsConfig:
    q0 = np.array(
        [
            0.0,
            -np.pi / 4,
            0.0,
            -3 * np.pi / 4,
            0.0,
            np.pi / 2,
            np.pi / 4,
        ]
    )

    kp_posture = 100.0
    posture_task_mask = np.array([1, 1, 1, 1, 1, 1, 0])
    joint_weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])


class InverseDynamicsController(Controller):
    def __init__(self, conf: InverseDynamicsConfig = None, dt: float = 0.001, **kwargs):
        super().__init__(**kwargs)
        self.conf = conf if conf is not None else InverseDynamicsConfig()
        self.robot = tsid.RobotWrapper(self.path_to_urdf, [], False)

    def update(self, q: np.array, dq: np.array) -> np.array:
        self._update_robot_model(q, dq)

        kp = self.conf.kp_posture * np.eye(self.model.nv)
        kd = 2.0 * np.sqrt(self.conf.kp_posture) * np.eye(self.model.nv)

        tau = self.data.M @ (kp @ (self._target_q - q) + kd @ (self._target_dq - dq)) + self.data.nle
        return tau

    def set_target(self, target_pose: pin.SE3 = None, target_q: np.array = None, target_dq: np.array = None):
        self._target_pose = target_pose
        self._target_q = target_q
        self._target_dq = target_dq
