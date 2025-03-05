"""Utility module for controller creation and logging functionality.

This module provides utilities for:
1. Creating different types of robot controllers (TSID, Cartesian Impedance, etc.)
2. Setting up throttled logging capabilities
"""

import logging
import time
from rich.logging import RichHandler
import numpy as np
from controllers.cartesian_impedance_controller import CartesianImpedanceConfig, CartesianImpedanceController  # noqa: F401
from controllers.controller import Controller, ControllerConfig
from controllers.gravity_compensation import GravityCompensationController  # noqa: F401
from controllers.tsid_controller import TSIDController, TSIDJointConfig, TSIDConfig  # noqa: F401
from controllers.inverse_dynamics_controller import InverseDynamicsController, InverseDynamicsConfig  # noqa: F401


def make_controller(ctrl_type: str, path_urdf: str, sim_dt: float) -> tuple[Controller, ControllerConfig]:
    """Create and configure a robot controller based on the specified type.

    Args:
        ctrl_type (str): Type of controller to create. Options are:
            - "tsid": Task Space Inverse Dynamics controller
            - "cartesian_impedance": Cartesian impedance controller
            - "gravity_compensation": Gravity compensation controller
            - "inverse_dynamics": Inverse dynamics controller
        path_urdf (str): Path to the URDF file describing the robot
        sim_dt (float): Simulation time step / controller update period

    Returns:
        tuple[Controller, ControllerConfig]: A tuple containing:
            - The instantiated controller object
            - The controller's configuration object

    Note:
        Each controller type comes with its specific configuration parameters.
        TSID controller has additional parameters for end-effector control and posture regulation.
    """
    # TODO: allow to load a config from a file
    if ctrl_type == "tsid":
        conf = TSIDConfig()

        conf.kp_ee = 1.0
        conf.ee_weights = np.array([200.0, 200.0, 200.0, 500.0, 500.0, 500.0])
        conf.kp_posture = 100.0
        conf.ctrl_freq = 1.0 / sim_dt

        ctrl = TSIDController(conf=conf, path_to_urdf=path_urdf)

    elif ctrl_type == "cartesian_impedance":
        conf = CartesianImpedanceConfig()
        ctrl = CartesianImpedanceController(conf=conf, path_to_urdf=path_urdf)
    elif ctrl_type == "gravity_compensation":
        conf = ControllerConfig()
        ctrl = GravityCompensationController(path_to_urdf=path_urdf)
    elif ctrl_type == "inverse_dynamics":
        conf = InverseDynamicsConfig()
        ctrl = InverseDynamicsController(conf=conf, path_to_urdf=path_urdf)

    return ctrl, conf


class ThrottleFilter(logging.Filter):
    """A logging filter that limits the frequency of log messages.

    This filter ensures that log messages are only emitted at a specified minimum
    time interval, preventing log flooding in high-frequency operations.

    Args:
        name (str, optional): Name of the filter. Defaults to "".
        interval (int, optional): Minimum time (in seconds) between log messages. Defaults to 5.
    """

    def __init__(self, name="", interval=5):
        super().__init__(name)
        self.interval = interval
        self.last_log_time = 0

    def filter(self, record):
        """Filter log records based on the time interval.

        Args:
            record: The log record to be filtered

        Returns:
            bool: True if enough time has passed since the last log message,
                  False otherwise
        """
        now = time.time()
        if now - self.last_log_time >= self.interval:
            self.last_log_time = now
            return True
        return False


def make_throttled_logger(name: str, interval: int):
    """Create a logger with throttled output using Rich formatting.

    Creates a logger that limits its output frequency and uses Rich for
    enhanced terminal formatting and display.

    Args:
        name (str): Name of the logger
        interval (int): Minimum time (in seconds) between log messages

    Returns:
        logging.Logger: Configured logger instance with throttling enabled

    Example:
        >>> logger = make_throttled_logger("robot_controller", 5)
        >>> logger.info("This message will appear at most every 5 seconds")
    """
    FORMAT = "%(message)s"
    logging.basicConfig(level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])

    logger = logging.getLogger(name)
    logger.addFilter(ThrottleFilter(name=name, interval=interval))
    return logger
