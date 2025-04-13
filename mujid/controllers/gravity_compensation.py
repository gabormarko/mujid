"""
Gravity Compensation Controller Module.

This module implements a simple gravity compensation controller that computes
and returns the gravity torques required to counteract the effects of gravity
on the robot's joints. This controller:
- Maintains the robot's current position without motion
- Allows for easy manual manipulation of the robot
- Serves as a safety baseline for other controllers

The controller uses Pinocchio's dynamics computations to obtain accurate
gravity compensation torques based on the robot's current configuration.
"""

import numpy as np
import pinocchio as pin
from typing_extensions import override

from mujid.controllers.controller import Controller


class GravityCompensationController(Controller):
    """A controller that compensates for gravitational forces on the robot.

    This controller computes the torques needed to counteract gravity at each joint,
    effectively making the robot "weightless" from the perspective of its motors.
    When active, the robot will:
    - Maintain its position without falling
    - Be easily moveable by external forces
    - Not actively move or track trajectories

    The controller uses the robot's dynamic model to compute accurate gravity
    compensation torques based on the current joint configuration.

    Inherits:
        Controller: Base controller class providing robot model functionality
    """

    @override
    def update(self, t: float, q: np.array, dq: np.array) -> np.array:
        """Compute gravity compensation torques for the current robot state.

        Updates the robot's dynamic model and returns the gravity compensation
        torques for the current configuration.

        Args:
            t (float): Current time (unused in this controller)
            q (np.array): Current joint positions, shape (n_joints,)
            dq (np.array): Current joint velocities, shape (n_joints,)

        Returns:
            np.array: Gravity compensation torques, shape (n_joints,)
                These torques, when applied to the joints, will counteract
                the effect of gravity on the robot.

        Note:
            The controller only uses the position (q) to compute gravity torques.
            Velocities (dq) are ignored as gravity compensation is configuration-dependent
            but not velocity-dependent.
        """
        self._update_robot_model(q, dq)
        return self._data.g

    @override
    def set_target(self, target_pose: pin.SE3, target_q: np.array, target_dq: np.array):
        return
