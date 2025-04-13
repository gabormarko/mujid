"""
Cartesian Impedance Controller implementation for robot control.

This module provides an implementation of a Cartesian impedance controller that operates
in task space (Cartesian coordinates) with a hierarchical structure:
- Primary task: Cartesian pose control of the end-effector
- Secondary task: Joint space control in the nullspace

The controller implements the following control law:
    τ = J^T(Kp_p * e + Kd_p * ė) + N(Kp_s(q_d - q) + Kd_s(dq_d - dq)) + g(q)
where:
    - J is the end-effector Jacobian
    - N is the nullspace projector
    - Kp_p, Kd_p are primary task gains
    - Kp_s, Kd_s are secondary task gains
    - g(q) is the gravity compensation term
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
        weights_primary (np.array, optional): Weight factors for primary task gains.
            If None, uniform weights are used.
        weights_secondary (np.array, optional): Weight factors for secondary task gains.
            If None, uniform weights are used.
    """

    kp_primary: float = 200.0
    kd_primary: float = None

    kp_secondary: float = 0.1
    kd_secondary: float = None

    nv = 7

    weights_primary: np.array(float) = None
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


class CartesianImpedanceController(Controller):
    """Cartesian Impedance Controller for hierarchical robot control.

    This controller implements a hierarchical control structure with:
    1. Primary task: Cartesian impedance control of the end-effector pose
    2. Secondary task: Joint space control in the nullspace
    3. Gravity compensation

    The control law combines these components:
        τ = J^T(Kp_p * e + Kd_p * ė) + N(Kp_s(q_d - q) + Kd_s(dq_d - dq)) + g(q)

    where J is the Jacobian, N is the nullspace projector, and g(q) is gravity compensation.
    """

    def __init__(
        self,
        path_to_urdf: str,
        conf: CartesianImpedanceConfig = None,
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
            1. Update the forward kinematics
            2. Compute the error between target and current end-effector pose
            3. Compute primary task torques (Cartesian impedance control)
            4. Compute secondary task torques in the nullspace (joint space control)
            5. Add gravity compensation

        Args:
            t (float): Current time
            q (np.array): Current joint positions
            dq (np.array): Current joint velocities

        Returns:
            np.array: Computed joint torques combining primary task, nullspace,
                     and gravity compensation terms
        """

        self._update_robot_model(q, dq)

        end_effector_pose = self._data.oMf[self._end_effector_frame_id]
        diff_pose = end_effector_pose.actInv(self._target_pose)

        error = np.zeros(6)
        error[:3] = diff_pose.translation
        error[3:] = pin.log3(diff_pose.rotation)

        J = pin.computeFrameJacobian(self._model, self._data, q, self._end_effector_frame_id, pin.LOCAL)

        tau_primary = J.T @ (self.conf.Kp_primary @ error - self.conf.Kd_primary @ J @ dq)

        nullspace_projector = np.eye(self._model.nv) - np.linalg.pinv(J) @ J
        tau_secondary = self.conf.Kp_secondary @ (self._target_q - q) - self.conf.Kd_secondary @ (self._target_dq - dq)
        tau_nullspace = nullspace_projector @ tau_secondary

        tau_gravity = pin.computeGeneralizedGravity(self._model, self._data, q)

        return tau_primary + tau_nullspace + tau_gravity
