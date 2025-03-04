from typing_extensions import override
from controllers.controller import Controller
import numpy as np


class GravityCompensationController(Controller):
    @override
    def update(self, q: np.array, dq: np.array) -> np.array:
        self._update_robot_model(q, dq)
        return self.data.g
