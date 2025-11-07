"""
Cartesian Impedance Controller implementation for robot control.

This module provides an implementation of a Cartesian impedance controller that operates
in task space (Cartesian coordinates).

It also provides with a default configuration class for the controller with some working default values.
"""

import numpy as np
import pinocchio as pin
from typing_extensions import override

from mujid.controllers.controller import Controller, ControllerConfig


class CartesianImpedanceConfig(ControllerConfig):
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
        weights_primary (np.ndarray, optional): Weight factors for primary task gains.
            If None, uniform weights are used.
        weights_secondary (np.ndarray, optional): Weight factors for secondary task gains.
            If None, uniform weights are used.
    """

    """
    task.k_pos_x: 5000.0
    task.k_pos_y: 5000.0
    task.k_pos_z: 5000.0
    task.k_rot_x: 200.0
    task.k_rot_y: 200.0
    task.k_rot_z: 200.0

    task.d_pos_x: 150.0
    task.d_pos_y: 150.0
    task.d_pos_z: 150.0
    task.d_rot_x: 4.0
    task.d_rot_y: 4.0
    task.d_rot_z: 4.0

    task.error_clip.rx: 0.08
    task.error_clip.ry: 0.08
    task.error_clip.rz: 0.08
    task.error_clip.x: 0.003
    task.error_clip.y: 0.003
    task.error_clip.z: 0.003"""

    kp_primary: float = 200.0
    kd_primary: float | None = 4.0

    kp_secondary: float = 5.0
    kd_secondary: float | None = None

    target_pose: pin.SE3 | None = None

    nv = 7

    weights_kp_primary: np.ndarray | None = np.array([25, 25, 25, 1, 1, 1])
    weights_kd_primary: np.ndarray | None = np.array([37.5, 37.5, 37.5, 1, 1, 1])
    weights_secondary: np.ndarray | None = None

    use_local_jacobian: bool = True

    @property
    def Kp_primary(self) -> np.ndarray:
        """Calculate the primary task proportional gain matrix.

        Returns:
            np.ndarray: 6x6 diagonal gain matrix for Cartesian space control
        """
        return self.Kp(n=6, kp=self.kp_primary, diag_weights=self.weights_kp_primary)

    @property
    def Kd_primary(self) -> np.ndarray:
        """Calculate the primary task derivative gain matrix.

        Returns:
            np.ndarray: 6x6 diagonal damping matrix for Cartesian space control
        """
        return self.Kd(
            n=6,
            kp=self.kp_primary,
            kd=self.kd_primary,
            diag_weights=self.weights_kd_primary,
        )

    @property
    def Kp_secondary(self) -> np.ndarray:
        """Calculate the secondary task proportional gain matrix.

        Returns:
            np.ndarray: nv x nv diagonal gain matrix for joint space control
        """
        return self.Kp(
            n=self.nv, kp=self.kp_secondary, diag_weights=self.weights_secondary
        )

    @property
    def Kd_secondary(self) -> np.ndarray:
        """Calculate the secondary task derivative gain matrix.

        Returns:
            np.ndarray: nv x nv diagonal damping matrix for joint space control
        """
        return self.Kd(
            n=self.nv,
            kp=self.kp_secondary,
            kd=self.kd_secondary,
            diag_weights=self.weights_secondary,
        )


class CartesianImpedanceController(Controller):
    """Cartesian Impedance Controller for hierarchical robot control.

    This controller implements a hierarchical control structure with:

    1. Primary task: Cartesian impedance control of the end-effector pose
    2. Secondary task: Joint space control in the nullspace
    3. Gravity compensation

    The controller implements the following control law:

    $$
    \\tau = \\tau_p + \\tau_{ns} + \\tau_g
    $$

    $$
    \\begin{cases}
    \\tau_p && = J^\\top(K_p^p e + K_d^p \\dot{e}) \\\\
    \\tau_{ns} && = N(K_p^s(q_d - q) + K_d^s(\\dot{q}_d - \\dot{q})) \\\\
    \\tau_g && = g(q) \\\\
    \\end{cases}
    $$

    where:

    - $\\tau_p$ is the primary task torque
    - $\\tau_{ns}$ is the secondary task torque in the nullspace
    - $\\tau_g$ is the gravity compensation torque

    The other terms are:

    - $J$ is the end-effector Jacobian
    - $N$ is the nullspace projector
    - $K_p^p, K_d^p$ are primary task gains
    - $K_p^s, K_d^s$ are secondary task gains
    - $g(q)$ is the gravity compensation term
    - $e\\in \\mathbb{R}^6$ is the pose error in the tangent space of $SE3$
    - $\\dot{e}$ is the velocity error in our case since we are not tracking velocities it is simply 
      the negative end-effector twist $-J \\dot{q}$
    - $q, \\dot{q}$ are joint positions and velocities
    - $q_d, \\dot{q}_d$ are desired joint positions and velocities for the secondary task
    """

    def __init__(
        self,
        path_to_urdf: str,
        conf: CartesianImpedanceConfig | None = None,
    ):
        """Initialize the Cartesian Impedance Controller.

        Args:
            path_to_urdf (str): Path to the robot's URDF file
            conf (CartesianImpedanceConfig, optional): Controller configuration.
                If None, default configuration is used.
        """
        super().__init__(path_to_urdf=path_to_urdf)
        self.conf = CartesianImpedanceConfig() if conf is None else conf
        self._end_effector_frame_id = self.conf.enf_effector_frame_id(self._model)

        self._target_pose = (
            conf.target_pose
            if conf is not None and conf.target_pose is not None
            else pin.SE3.Identity()
        )

        self._target_q = pin.neutral(self._model)
        self._target_dq = np.zeros(self.conf.nv)

    @override
    def set_target(
        self,
        target_pose: pin.SE3 | None = None,
        target_q: np.ndarray | None = None,
        target_dq: np.ndarray | None = None,
    ):
        """Set the target pose, joint position and joint velocity.

        Args:
            target_pose (pin.SE3, optional): Target end-effector pose in SE3
            target_q (np.ndarray, optional): Target joint positions.
                If None, uses configuration default.
            target_dq (np.ndarray, optional): Target joint velocities.
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
            self._target_dq = np.zeros(self.conf.nv)

    @override
    def update(self, t: float, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        self._update_robot_model(q, dq)

        end_effector_pose = self._data.oMf[self._end_effector_frame_id]  # type: ignore
        diff_pose = (
            end_effector_pose.actInv(self._target_pose)
            if self.conf.use_local_jacobian
            else self._target_pose.act(end_effector_pose.inverse())
        )

        error = pin.log(diff_pose)  # project to tangent space of SE3 # type: ignore

        J = pin.computeFrameJacobian(
            self._model,
            self._data,
            q,
            self._end_effector_frame_id,
            pin.LOCAL if self.conf.use_local_jacobian else pin.WORLD,
        )

        tau_primary = J.T @ (
            self.conf.Kp_primary @ error - self.conf.Kd_primary @ J @ dq
        )

        nullspace_projector = np.eye(self._model.nv) - np.linalg.pinv(J) @ J  # type: ignore
        tau_secondary = self._data.M @ (
            self.conf.Kp_secondary @ (self._target_q - q)
            + self.conf.Kd_secondary @ (self._target_dq - dq)
        )  # type: ignore
        tau_nullspace = nullspace_projector @ tau_secondary

        tau_gravity = pin.computeGeneralizedGravity(self._model, self._data, q)

        return tau_primary + tau_nullspace + tau_gravity
