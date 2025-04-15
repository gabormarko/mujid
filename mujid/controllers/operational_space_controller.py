"""
Operational Space Controller implementation for robot control.

This module provides an implementation of an operational space controller that operates
in task space (Cartesian coordinates) with dynamic consistency.

It also provides a default configuration class for the controller with some working default values.
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

    use_local_jacobian: bool = True

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

    This controller implements a dynamically consistent control structure with:

    1. Primary task: Operational space control of the end-effector pose
    2. Secondary task: Dynamically consistent nullspace control
    3. Dynamic compensation (inertia, Coriolis, gravity)

    The controller implements the following control law:

    $$
    \\tau = \\tau_p + \\tau_{ns} + \\tau_d
    $$

    $$
    \\begin{cases}
    \\tau_p && = J^\\top \\Lambda(x) f^*(x) \\\\
    \\tau_{ns} && = N^\\top(q)[K_p^s(q_d - q) + K_d^s(\\dot{q}_d - \\dot{q})] \\\\
    \\tau_d && = h(q,\\dot{q}) \\\\
    \\end{cases}
    $$

    where:

    - $\\tau_p$ is the primary task torque
    - $\\tau_{ns}$ is the dynamically consistent nullspace torque
    - $\\tau_d$ is the dynamic compensation torque

    The other terms are:

    - $\\Lambda(x) = (J M^{-1} J^\\top)^{-1}$ is the operational space inertia matrix
    - $f^*(x) = K_p^p e + K_d^p \\dot{e}$ is the task space control force
    - $N^\\top(q) = I - J^\\top J_b^\\top$ is the dynamically consistent nullspace projector
    - $J_b = M^{-1} J^\\top \\Lambda$ is the dynamically consistent inverse of $J$
    - $h(q,\\dot{q}) = C(q,\\dot{q})\\dot{q} + g(q)$ is the dynamic compensation term
    - $e\\in \\mathbb{R}^6$ is the pose error in the tangent space of $SE3$
    - $\\dot{e}$ is the velocity error in task space
    - $q, \\dot{q}$ are joint positions and velocities
    - $q_d, \\dot{q}_d$ are desired joint positions and velocities for the secondary task
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

        Implements the operational space control law:

        $$
        \\tau = \\underbrace{J^\\top \\Lambda(x) f^*(x)}_{\\text{primary task}} +
                \\underbrace{N^\\top(q)[K_p^s(q_d - q) + K_d^s(\\dot{q}_d - \\dot{q})]}_{\\text{nullspace task}} +
                \\underbrace{h(q,\\dot{q})}_{\\text{dynamics}}
        $$

        Args:
            t (float): Current time
            q (np.array): Current joint positions ($q$)
            dq (np.array): Current joint velocities ($\\dot{q}$)

        Returns:
            np.array: Computed joint torques ($\\tau$) combining primary task, nullspace,
                     and dynamic compensation terms
        """
        self._update_robot_model(q, dq)

        # Compute end-effector pose error
        end_effector_pose = self._data.oMf[self._end_effector_frame_id]
        diff_pose = (
            end_effector_pose.actInv(self._target_pose)
            if self.conf.use_local_jacobian
            else self._target_pose.act(end_effector_pose.inverse())
        )
        error = pin.log(diff_pose)  # project to tangent space of SE3

        # Compute Jacobian and operational space quantities
        J = pin.computeFrameJacobian(
            self._model,
            self._data,
            q,
            self._end_effector_frame_id,
            pin.LOCAL if self.conf.use_local_jacobian else pin.WORLD,
        )
        M_inv = np.linalg.inv(self._data.M)
        Lambda = np.linalg.pinv(J @ M_inv @ J.T)  # operational space inertia matrix
        J_bar = M_inv @ J.T @ Lambda  # dynamically consistent inverse

        # Primary task: operational space control
        f_star = self.conf.Kp_primary @ error - self.conf.Kd_primary @ (J @ dq)
        tau_primary = J.T @ Lambda @ f_star

        # Secondary task: dynamically consistent nullspace control
        N_bar = np.eye(self._model.nv) - J.T @ J_bar.T  # dynamically consistent nullspace projector
        tau_secondary = self.conf.Kp_secondary @ (self._target_q - q) + self.conf.Kd_secondary @ (self._target_dq - dq)
        tau_nullspace = N_bar @ tau_secondary

        # Dynamic compensation (includes Coriolis and gravity)
        tau_dynamic = self._data.nle

        return tau_primary + tau_nullspace + tau_dynamic
