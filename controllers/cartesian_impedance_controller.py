from typing_extensions import override
from controllers.controller import Controller
import numpy as np
import pinocchio as pin


class CartesianImpedanceController(Controller):
    def __init__(
        self,
        *,
        target_frame_name: str,
        Kp: float | np.array(float) = 100.0,
        Kd: float | np.array(float) = 2 * np.sqrt(100.0),
        Kp_s: float | np.array(float) = 0.1,
        Kd_s: float | np.array(float) = 2 * np.sqrt(0.1),
        target_pose: pin.SE3 = None,
        **kwargs,
    ):
        """Initialize the Cartesian Impedance Controller.

        Args:
            target_frame_name (str): The name of the target frame.
            Kp (float | np.array(float), optional): The proportional gain. Defaults to 100.0.
            Kd (float | np.array(float), optional): The derivative gain. Defaults to 2 * np.sqrt(100.0).
            Kp_s (float | np.array(float), optional): The secondary task gain. Defaults to 0.0.
            Kd_s (float | np.array(float), optional): The secondary task derivative gain. Defaults to 0.0.
            target_pose (pin.SE3, optional): The target pose. Defaults to None.
            kwargs: Additional arguments.
        """
        super().__init__(**kwargs)
        self.end_effector_frame = target_frame_name
        self.end_effector_frame_id = self.model.getFrameId(self.end_effector_frame)

        self.Kp = Kp if isinstance(Kp, np.ndarray) else np.eye(6) * Kp
        self.Kd = Kd if isinstance(Kd, np.ndarray) else np.eye(6) * Kd

        self.Kp_s = Kp_s if isinstance(Kp_s, np.ndarray) else np.eye(self.model.nv) * Kp_s
        self.Kd_s = Kd_s if isinstance(Kd_s, np.ndarray) else np.eye(self.model.nv) * Kd_s

        self.target_pose = target_pose if target_pose is not None else pin.SE3.Identity()
        self.target_q = pin.neutral(self.model)
        self.target_dq = np.zeros(self.model.nv)

    @override
    def set_target(self, target_pose: pin.SE3 = None, target_q: np.array = None, target_dq: np.array = None):
        """Set the target pose, joint position and joint velocity."""
        if target_pose is not None:
            self.target_pose = target_pose

        if target_q is not None:
            self.target_q = target_q
        else:
            self.target_q = pin.neutral(self.model)

        if target_dq is not None:
            self.target_dq = target_dq
        else:
            self.target_dq = np.zeros(self.model.nv)

    @override
    def update(self, q: np.array, dq: np.array) -> np.array:
        """Update step for the controller.

        The steps for the controller are:
            1. Update the forward kinematics.
            2. Comute the error between the target pose and the current pose.
            3. Compute the torque based on the error.
            4. Compute the torque in the nullspace for the secondary task, which is following the target joint position and velocity.
            5. Compute the gravity compensation torque.
        """

        self._update_robot_model(q, dq)

        end_effector_pose = self.data.oMf[self.end_effector_frame_id]
        diff_pose = end_effector_pose.actInv(self.target_pose)

        error = np.zeros(6)
        error[:3] = diff_pose.translation
        error[3:] = pin.log3(diff_pose.rotation)

        J = pin.computeFrameJacobian(self.model, self.data, q, self.end_effector_frame_id, pin.LOCAL)

        tau_task = J.T @ (self.Kp @ error - self.Kd @ J @ dq)

        nullspace_projector = np.eye(self.model.nv) - np.linalg.pinv(J) @ J
        tau_secondary = self.Kp_s @ (self.target_q - q) - self.Kd_s @ (self.target_dq - dq)
        tau_nullspace = nullspace_projector @ tau_secondary

        tau_gravity = pin.computeGeneralizedGravity(self.model, self.data, q)

        return tau_task + tau_nullspace + tau_gravity
