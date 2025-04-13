"""
Operational Space Controller implementation for robot control.

This module provides an implementation of an operational space controller that operates
in task space (Cartesian coordinates) with dynamic consistency:
- Primary task: Cartesian pose control of the end-effector using operational space formulation
- Secondary task: Dynamically consistent nullspace control
- Dynamic compensation: Full robot dynamics compensation

The controller implements the following control law:
    τ = J^T Λ(x) f*(x) + N^T(q)[Kp_s(q_d - q) + Kd_s(dq_d - dq)] + h(q,dq)
where:
    - Λ(x) = (J M^{-1} J^T)^{-1} is the operational space inertia matrix
    - f*(x) = Kp * e + Kd * ė is the task space control force
    - N^T(q) is the dynamically consistent nullspace projector
    - h(q,dq) = C(q,dq)dq + g(q) is the dynamic compensation term
"""

import numpy as np
import pinocchio as pin
from typing_extensions import override

from mujid.controllers.controller import Controller, ControllerConfig


class OperationalSpaceControllerConfig(ControllerConfig):
    """Configuration class for the Cartesian Impedance Controller.

    Attributes:
        kp_primary (float): Proportional gain for the primary task (Cartesian space).
            Defaults to 200.0.
        kd_primary (float, optional): Derivative gain for the primary task.
            If None, computed automatically for critical damping.
        kp_secondary (float): Proportional gain for the secondary task (joint space).
            Defaults to 0.1.
        kd_secondary (float, optional): Derivative gain for the secondary task.
            If None, computed automatically for critical damping.
        nv (int): Number of robot joints (degrees of freedom). Defaults to 7.
        weights_primary (np.array, optional): Weight factors for primary task gains.
            If None, uniform weights are used.
        weights_secondary (np.array, optional): Weight factors for secondary task gains.
            If None, uniform weights are used.
    """

    kp_primary: float = 200.0
    kd_primary: float = None

    kp_secondary: float = 2.0
    kd_secondary: float = None

    nv = 7

    weights_primary: np.array(float) = [1.0, 1.0, 1.0, 10.0, 10.0, 10.0]
    weights_secondary: np.array(float) = None

    @property
    def Kp_primary(self) -> np.array:
        """Calculate the primary task proportional gain matrix.

        Returns:
            np.array: 6x6 diagonal gain matrix for Cartesian space control
        """
        return self.Kp(n=6, kp=self.kp_primary, diag_weights=self.weights_primary)

    @property
    def Kd_primary(self) -> np.array:
        """Calculate the primary task derivative gain matrix.

        Returns:
            np.array: 6x6 diagonal damping matrix for Cartesian space control
        """
        return self.Kd(n=6, kp=self.kp_primary, kd=self.kd_primary, diag_weights=self.weights_primary)

    @property
    def Kp_secondary(self) -> np.array:
        """Calculate the secondary task proportional gain matrix.

        Returns:
            np.array: nv x nv diagonal gain matrix for joint space control
        """
        return self.Kp(n=self.nv, kp=self.kp_secondary, diag_weights=self.weights_secondary)

    @property
    def Kd_secondary(self) -> np.array:
        """Calculate the secondary task derivative gain matrix.

        Returns:
            np.array: nv x nv diagonal damping matrix for joint space control
        """
        return self.Kd(n=self.nv, kp=self.kp_secondary, kd=self.kd_secondary, diag_weights=self.weights_secondary)


class OperationalSpaceController(Controller):
    """Operational Space Controller for dynamically consistent robot control.

    This controller implements operational space control with:
    1. Primary task: Operational space control of the end-effector pose
    2. Secondary task: Dynamically consistent nullspace control
    3. Dynamic compensation (inertia, Coriolis, gravity)

    The control law implements the operational space formulation:
        τ = J^T Λ(x) f*(x) + N^T(q)[Kp_s(q_d - q) + Kd_s(dq_d - dq)] + h(q,dq)

    where Λ(x) is the operational space inertia matrix, f*(x) is the task space control force,
    and h(q,dq) provides direct dynamic compensation in joint space.
    """

    def __init__(
        self,
        path_to_urdf: str,
        conf: OperationalSpaceControllerConfig = None,
    ):
        """Initialize the Operational Space Controller.

        Args:
            path_to_urdf (str): Path to the robot's URDF file
            conf (OperationalSpaceConfig, optional): Controller configuration.
                If None, default configuration is used.
        """
        super().__init__(path_to_urdf=path_to_urdf)
        self.conf = OperationalSpaceControllerConfig() if conf is None else conf
        self._end_effector_frame_id = self.conf.enf_effector_frame_id(self._model)

        self._target_pose = conf.target_pose if conf.target_pose is not None else pin.SE3.Identity()

        self._target_q = pin.neutral(self._model)
        self._target_dq = np.zeros(self._model.nv)

    @override
    def set_target(self, target_pose: pin.SE3 = None, target_q: np.array = None, target_dq: np.array = None):
        """Set the target pose, joint position and joint velocity.

        Args:
            target_pose (pin.SE3, optional): Target end-effector pose in SE3
            target_q (np.array, optional): Target joint positions.
                If None, uses configuration default.
            target_dq (np.array, optional): Target joint velocities.
                If None, sets zero velocity.
        """
        if target_pose is not None:
            self._target_pose = target_pose

        if target_q is not None:
            self._target_q = target_q
        else:
            self._target_q = self.conf.q0

        if target_dq is not None:
            self._target_dq = target_dq
        else:
            self._target_dq = np.zeros(self._model.nv)

    @override
    def update(self, t: float, q: np.array, dq: np.array) -> np.array:
        """Compute control torques based on current robot state.

        The controller performs the following steps:
            1. Update the robot dynamics (M, C, g)
            2. Compute operational space quantities (Λ)
            3. Compute the error between target and current end-effector pose
            4. Compute primary task forces in operational space
            5. Project to joint space with dynamic consistency
            6. Add dynamically consistent nullspace control
            7. Add direct dynamic compensation

        Args:
            t (float): Current time
            q (np.array): Current joint positions
            dq (np.array): Current joint velocities

        Returns:
            np.array: Computed joint torques with dynamic consistency
        """
        # Update robot dynamics
        self._update_robot_model(q, dq)

        # Get mass matrix and compute its inverse
        M_inv = np.linalg.inv(self._data.M)

        # Compute task space quantities
        J = pin.computeFrameJacobian(self._model, self._data, q, self._end_effector_frame_id, pin.LOCAL)

        # Operational space inertia matrix (Λ)
        Lambda = np.linalg.pinv(J @ M_inv @ J.T)

        # Dynamically consistent inverse of J
        J_bar = M_inv @ J.T @ Lambda

        # Compute task space error
        end_effector_pose = self._data.oMf[self._end_effector_frame_id]
        diff_pose = end_effector_pose.actInv(self._target_pose)

        error = np.zeros(6)
        error[:3] = diff_pose.translation
        error[3:] = pin.log3(diff_pose.rotation)

        # Compute operational space force
        f_star = self.conf.Kp_primary @ error - self.conf.Kd_primary @ (J @ dq)

        # Primary task torques with dynamic consistency
        tau_primary = J.T @ Lambda @ (f_star)

        # Dynamic compensation torques
        tau_dynamic = self._data.nle

        # Dynamically consistent nullspace projector
        N_bar = np.eye(self._model.nv) - J.T @ J_bar.T

        # Nullspace control (joint space)
        tau_secondary = self.conf.Kp_secondary @ (self._target_q - q) - self.conf.Kd_secondary @ (dq - self._target_dq)
        tau_nullspace = N_bar @ tau_secondary

        return tau_primary + tau_nullspace + tau_dynamic
