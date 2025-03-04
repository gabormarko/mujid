import pinocchio as pin
import numpy as np
from abc import ABC, abstractmethod


class Controller(ABC):
    """Simple controller class for inverse dynamics control."""

    def __init__(self, path_to_urdf: str):
        """Initialize the controller with the path to the URDF file."""
        self.path_to_urdf = path_to_urdf
        self.model = pin.buildModelFromUrdf(path_to_urdf)
        self.data = self.model.createData()

    def _update_robot_model(self, q: np.array, dq: np.array):
        pin.forwardKinematics(self.model, self.data, q, dq)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeAllTerms(self.model, self.data, q, dq)

    @abstractmethod
    def update(self, q: np.array, dq: np.array) -> np.array:
        """Update the state of the controller and return the desired torques."""
        raise NotImplementedError

    def set_target(self, target_pose: pin.SE3) -> np.array:
        """Update the state of the controller and return the desired torques."""
        raise NotImplementedError
