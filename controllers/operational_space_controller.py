"""TODO"""

import numpy as np
import pinocchio as pin
from dataclasses import dataclass
from controllers.controller import Controller


@dataclass
class OperationalSpaceControllerConfig:
    target_frame_name: str = "fr3_hand_tcp"
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

    kp = 0.01
    kp_posture = 0.0
    joint_weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])


class OperationalSpaceController(Controller):
    def __init__(self, conf: OperationalSpaceControllerConfig = None, dt: float = 0.001, **kwargs):
        raise NotImplementedError("OperationalSpaceController is not working yet.")

        super().__init__(**kwargs)
        self.conf = conf if conf is not None else OperationalSpaceControllerConfig()
        self.end_effector_frame_id = self._model.getFrameId(self.conf.target_frame_name)

    def update(self, q: np.array, dq: np.array) -> np.array:
        self._update_robot_model(q, dq)

        end_effector_pose = self._data.oMf[self.end_effector_frame_id]
        diff_pose = end_effector_pose.actInv(self._target_pose)

        error = np.zeros(6)
        error[:3] = diff_pose.translation
        error[3:] = pin.log3(diff_pose.rotation)

        J = pin.computeFrameJacobian(self._model, self._data, q, self.end_effector_frame_id, pin.LOCAL)
        pin.computeJointJacobiansTimeVariation(self._model, self._data, q, dq)
        dJ = pin.getFrameJacobianTimeVariation(self._model, self._data, self.end_effector_frame_id, pin.LOCAL)

        kp = self.conf.kp * np.eye(6)
        kd = 2.0 * np.sqrt(self.conf.kp) * np.eye(6)

        ddX_target = kp @ error + kd @ J @ dq

        Minv = pin.computeMinverse(self._model, self._data, q)
        J_T_pinv = np.linalg.pinv(J.T)
        J_pinv = np.linalg.pinv(J)

        Lambda = J_T_pinv @ Minv @ J_pinv
        mu = J_T_pinv @ self._data.nle - Lambda @ dJ @ dq

        f = Lambda @ ddX_target + mu
        tau = J.T @ f

        nullspace_projector = np.eye(self._model.nv) - J_pinv @ J
        kp_s = self.conf.kp_posture * np.eye(self._model.nv)
        kd_s = 2.0 * np.sqrt(self.conf.kp_posture) * np.eye(self._model.nv)
        tau_secondary = self._data.M @ (kp_s @ (self._target_q - q) + kd_s @ (self._target_dq - dq)) + self._data.nle
        tau_nullspace = nullspace_projector @ tau_secondary

        return tau + tau_nullspace

    def set_target(self, target_pose: pin.SE3 = None, target_q: np.array = None, target_dq: np.array = None):
        self._target_pose = target_pose
        self._target_q = target_q
        self._target_dq = target_dq
