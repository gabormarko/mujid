"""
Base Controller Module for Robot Control.

This module provides base classes and utilities for implementing robot controllers:
- ControllerConfig: Base configuration dataclass for controller parameters
- Controller: Abstract base class for robot controllers

The module implements common functionality such as:
- Robot model initialization using Pinocchio
- Forward kinematics and dynamics updates
- Gain matrix calculations with optional diagonal weighting
- Standard controller interface definitions
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pinocchio as pin


@dataclass()
class ControllerConfig:
    """Base configuration class for robot controllers.

    This dataclass provides common configuration parameters and utility methods
    for robot controllers.

    Attributes:
        ctrl_freq (float): Control frequency in Hz. Defaults to 500.0.
        end_effector_frame (str): Name of the end-effector frame in the URDF.
            Defaults to "fr3_hand_tcp".
        target_pose (pin.SE3, optional): Target pose for the end-effector.
        q0 (np.array): Default initial joint configuration. Shape (7,) for FR3 robot.
    """

    ctrl_freq: float = 500.0
    end_effector_frame: str = "fr3_hand_tcp"
    target_pose: pin.SE3 | None = None

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

    @property
    def ctrl_dt(self) -> float:
        """Calculate the control time step from the control frequency.

        Returns:
            float: Control period in seconds (1/frequency)
        """
        return 1.0 / self.ctrl_freq

    def enf_effector_frame_id(self, model):
        """Get the frame ID for the end-effector from the robot model.

        Args:
            model: Pinocchio robot model

        Returns:
            int: Frame ID of the end-effector in the model
        """
        return model.getFrameId(self.end_effector_frame)

    @staticmethod
    def Kp(n: int, kp: float, diag_weights: np.ndarray | None = None) -> np.ndarray:
        """Compute proportional gain matrix with optional diagonal weighting.

        Args:
            n (int): Size of the gain matrix
            kp (float): Base proportional gain value
            diag_weights (np.array, optional): Diagonal weights for scaling gains.
                If provided, must have length n.

        Returns:
            np.array: n x n diagonal gain matrix

        Raises:
            AssertionError: If diag_weights length doesn't match n
        """
        if diag_weights is not None:
            assert len(diag_weights) == n, "Invalid number of weights."
            return kp * np.diag(diag_weights)
        return kp * np.eye(n)

    @staticmethod
    def Kd(
        n: int,
        kp: float,
        kd: float | None = None,
        diag_weights: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute derivative gain matrix with optional diagonal weighting.

        If kd is not provided, it's computed for critical damping as:
            kd = 2 * sqrt(kp)

        Args:
            n (int): Size of the gain matrix
            kp (float): Proportional gain value (used if kd is None)
            kd (float, optional): Base derivative gain value.
                If None, computed for critical damping.
            diag_weights (np.array, optional): Diagonal weights for scaling gains.
                If provided, must have length n.

        Returns:
            np.array: n x n diagonal damping matrix

        Raises:
            AssertionError: If diag_weights length doesn't match n
        """
        if kd is None:
            kd = 2.0 * np.sqrt(kp)

        if diag_weights is not None:
            assert len(diag_weights) == n, "Invalid number of weights."
            if kd is not None:
                return kd * np.diag(diag_weights)
            else:
                return 2 * np.sqrt(kp * np.diag(diag_weights))
        else:
            if kd is not None:
                return kd * np.eye(n)
            else:
                return 2 * np.sqrt(kp) * np.eye(n)


class Controller(ABC):
    """Abstract base class for robot controllers.

    This class provides the basic structure and common functionality for
    implementing robot controllers, including:
    - Robot model initialization using Pinocchio
    - Forward kinematics and dynamics updates
    - Standard interface for control updates and target setting

    The class uses Pinocchio for rigid body dynamics computations and
    requires a URDF model of the robot.

    Attributes:
        _path_to_urdf (str): Path to the robot's URDF file
        _model: Pinocchio robot model
        _data: Pinocchio robot data for computations
    """

    def __init__(self, path_to_urdf: str):
        """Initialize the controller with a robot model.

        Args:
            path_to_urdf (str): Path to the robot's URDF file
        """
        self._path_to_urdf = path_to_urdf
        self._model = pin.buildModelFromUrdf(path_to_urdf)
        self._data = self._model.createData()

    def _update_robot_model(self, q: np.ndarray, dq: np.ndarray):
        """Update the robot's kinematic and dynamic model.

        Computes:
        - Forward kinematics
        - Frame placements
        - All terms of the dynamic model (M, C, g)

        Args:
            q (np.array): Joint positions
            dq (np.array): Joint velocities
        """
        pin.forwardKinematics(self._model, self._data, q, dq)
        pin.updateFramePlacements(self._model, self._data)
        pin.computeAllTerms(self._model, self._data, q, dq)

    @abstractmethod
    def update(self, t: float, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """Compute control commands based on current robot state.

        This method should be implemented by concrete controller classes
        to compute joint torques or other control commands.

        Args:
            t (float): Current time in seconds
            q (np.array): Current joint positions
            dq (np.array): Current joint velocities

        Returns:
            np.array: Computed joint torques or control commands

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

    def set_target(
        self,
        target_pose: pin.SE3 | None = None,
        target_q: np.ndarray | None = None,
        target_dq: np.ndarray | None = None,
    ) -> np.ndarray:
        """Set target state for the controller.

        Args:
            target_pose (pin.SE3): Target end-effector pose
            q (np.array): Target joint positions
            dq (np.array): Target joint velocities

        Returns:
            np.array: Optional initial control commands

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError
