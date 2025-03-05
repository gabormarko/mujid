"""
Inverse Dynamics Controller implementation for robot control.

This module provides an implementation of an inverse dynamics controller, which computes
joint torques based on the robot's dynamic model and desired joint positions/velocities.
The controller uses the equation:
    τ = M(q)(Kp(q_desired - q) + Kd(dq_desired - dq)) + h(q,dq)
where:
    - M(q) is the mass matrix
    - h(q,dq) contains Coriolis, centrifugal, and gravity terms
    - Kp, Kd are gain matrices
"""

from typing_extensions import override
import tsid
import numpy as np
import pinocchio as pin
from dataclasses import dataclass
from controllers.controller import Controller, ControllerConfig


@dataclass
class InverseDynamicsConfig(ControllerConfig):
    """Configuration class for the Inverse Dynamics Controller.

    Attributes:
        kp_posture (float): Base proportional gain value. Defaults to 100.0.
        joint_weights (np.array, optional): Weight factors for each joint to adjust individual joint gains.
            If None, uniform weights are used.
    """

    kp_joints = 100.0
    kd_joints = None
    joint_weights: np.array = None

    nv = 7

    @property
    def Kp_joints(self) -> np.array:
        """Calculate the proportional gain matrix.

        Args:
            nv (int): Number of robot joints (degrees of freedom)

        Returns:
            np.array: Diagonal proportional gain matrix (nv x nv)
        """
        return self.Kp(n=self.nv, kp=self.kp_joints, diag_weights=self.joint_weights)

    @property
    def Kd_joints(self):
        """Calculate the derivative gain matrix.

        The derivative gains are computed as 2*sqrt(Kp) to achieve critical damping.

        Args:
            nv (int): Number of robot joints (degrees of freedom)

        Returns:
            np.array: Diagonal derivative gain matrix (nv x nv)
        """
        return self.Kd(n=self.nv, kp=self.kp_joints, kd=self.kd_joints, diag_weights=self.joint_weights)


class InverseDynamicsController(Controller):
    """Inverse Dynamics Controller for robot motion control.

    This controller computes joint torques using the robot's dynamic model and desired joint positions/velocities.
    It implements a PD control law in task space with gravity compensation:
        τ = M(q)(Kp(q_desired - q) + Kd(dq_desired - dq)) + h(q,dq)

    The controller automatically handles gravity compensation and accounts for the robot's
    full dynamics through the mass matrix M(q) and nonlinear effects h(q,dq).
    """

    def __init__(self, conf: InverseDynamicsConfig = None, **kwargs):
        """Initialize the Inverse Dynamics Controller.

        Args:
            conf (InverseDynamicsConfig, optional): Controller configuration parameters.
                If None, default configuration is used.
            **kwargs: Additional arguments passed to the parent Controller class.
        """
        super().__init__(**kwargs)
        self.conf = conf if conf is not None else InverseDynamicsConfig()
        self.robot = tsid.RobotWrapper(self._path_to_urdf, [], False)

    @override
    def update(self, t: float, q: np.array, dq: np.array) -> np.array:
        """Compute control torques based on current robot state.

        Args:
            q (np.array): Current joint positions
            dq (np.array): Current joint velocities

        Returns:
            np.array: Computed joint torques to achieve desired motion
        """
        self._update_robot_model(q, dq)

        tau = (
            self._data.M @ (self.conf.Kp_joints @ (self._target_q - q) + self.conf.Kd_joints @ (self._target_dq - dq))
            + self._data.nle
        )
        return tau

    @override
    def set_target(self, target_pose: pin.SE3 = None, target_q: np.array = None, target_dq: np.array = None):
        """Set target positions and velocities for the controller.

        Args:
            target_pose (pin.SE3, optional): Target end-effector pose in SE3
            target_q (np.array, optional): Target joint positions
            target_dq (np.array, optional): Target joint velocities
        """
        self._target_pose = target_pose
        self._target_q = target_q
        self._target_dq = target_dq
