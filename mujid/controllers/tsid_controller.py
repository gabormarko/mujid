"""
Task Space Inverse Dynamics (TSID) Controller Module.

This module implements a whole-body controller using Task Space Inverse Dynamics,
which allows for:
- Multi-task prioritized control
- End-effector tracking in Cartesian space
- Joint space posture control
- Dynamic constraint handling (joint limits, torque bounds)
- Hierarchical quadratic programming (HQP) optimization

The controller uses Pinocchio for rigid body dynamics and TSID for the
optimization-based inverse dynamics control framework.
"""

from dataclasses import dataclass

import numpy as np
import pinocchio as pin
import tsid
from typing_extensions import override

from mujid.controllers.controller import Controller, ControllerConfig


@dataclass
class TSIDConfig(ControllerConfig):
    """Configuration class for the TSID controller.

    This class defines all the parameters needed to configure the Task Space
    Inverse Dynamics controller, including task weights, gains, and constraints.

    Attributes:
        w_ee (float): Weight for end-effector task (default: 1.0)
        w_posture (float): Weight for posture task (default: 1e-3)
        w_torque_bounds (float): Weight for torque bounds constraint (default: 1.0)
        w_joint_bounds (float): Weight for joint bounds constraint (default: 1.0)
        nv (int): Number of robot joints/degrees of freedom (default: 7)
        kp_ee (float): Proportional gain for end-effector task (default: 100.0)
        kd_ee (float): Derivative gain for end-effector task (default: None, computed from kp)
        ee_weights (np.array): Optional weights for each end-effector DoF
        ee_task_mask (np.array): Mask for enabling/disabling end-effector DoFs
        ee_task_local_frame (bool): Whether to use local frame for end-effector task
        kp_posture (float): Proportional gain for posture task (default: 100.0)
        kd_posture (float): Derivative gain for posture task
        posture_weights (np.array): Optional weights for each joint in posture task
        posture_task_mask (np.array): Mask for enabling/disabling joints in posture task
        tau_max_scaling (float): Scaling factor for maximum joint torques (default: 0.4)
        v_max_scaling (float): Scaling factor for maximum joint velocities (default: 0.8)
    """

    w_ee = 1.0
    w_posture = 1e-3
    w_torque_bounds = 1.0
    w_joint_bounds = 1.0

    nv = 7

    kp_ee: float = 100.0
    kd_ee: float = None
    ee_weights: np.array = None
    ee_task_mask = np.ones(6)
    ee_task_local_frame = True
    ee_task_mask: np.array = None

    kp_posture = 100.0
    kd_posture = None
    posture_weights: np.array = None
    posture_task_mask: np.array = None

    tau_max_scaling = 0.4
    v_max_scaling = 0.8

    @property
    def Kp_ee(self) -> np.array:
        return np.diag(self.Kp(n=6, kp=self.kp_ee, diag_weights=self.ee_weights))

    @property
    def Kd_ee(self) -> np.array:
        return np.diag(self.Kd(n=6, kp=self.kp_ee, kd=self.kd_ee, diag_weights=self.ee_weights))

    @property
    def Kp_posture(self) -> np.array:
        return np.diag(self.Kp(n=self.nv, kp=self.kp_posture, diag_weights=self.posture_weights))

    @property
    def Kd_posture(self) -> np.array:
        return np.diag(self.Kd(n=self.nv, kp=self.kp_posture, kd=self.kd_posture, diag_weights=self.posture_weights))


class TSIDJointConfig(TSIDConfig):
    """Configuration for pure joint space control using TSID.

    This configuration disables end-effector tracking (w_ee = 0) and focuses on
    joint space control with specific weights for different joints.

    Attributes:
        w_ee (float): Set to 0.0 to disable end-effector tracking
        kp_posture (float): Higher gain for precise joint control
        posture_weights (np.array): Per-joint weights, with higher weight on the last joint
    """

    w_ee = 0.0
    kp_posture = 100.0
    posture_weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 20.0])


class TSIDController(Controller):
    """Task Space Inverse Dynamics Controller implementation.

    This controller uses a hierarchical quadratic programming approach to generate
    joint torques that achieve multiple prioritized tasks while respecting various
    constraints. It can handle:
    - Cartesian space end-effector control
    - Joint space posture control
    - Torque and joint velocity limits
    - Task prioritization through weights

    The controller solves an optimization problem at each time step to compute
    the optimal joint torques that achieve the desired tasks while minimizing
    a weighted sum of task errors.
    """

    def __init__(self, conf: TSIDConfig = None, **kwargs):
        super().__init__(**kwargs)
        self.conf = conf if conf is not None else TSIDConfig()
        self.robot = tsid.RobotWrapper(self._path_to_urdf, [], False)
        self._n_calls = 0
        self.initialize_tsid()

    @override
    def update(self, t: float, q: np.array, dq: np.array) -> np.array:
        """Compute control torques for the current robot state.

        Solves the hierarchical quadratic program to compute optimal joint torques
        that achieve the specified tasks while respecting constraints.

        Args:
            t (float): Current time
            q (np.array): Current joint positions, shape (n_joints,)
            dq (np.array): Current joint velocities, shape (n_joints,)

        Returns:
            np.array: Computed joint torques, shape (n_joints,)
                Returns zero torques if the QP solver fails.

        Note:
            The controller prioritizes tasks based on their weights and hierarchical levels.
            The QP solver status is checked to ensure a valid solution was found.
        """
        HQPData = self.formulation.computeProblemData(t, q, dq)

        sol = self.solver.solve(HQPData)
        if sol.status != 0:
            # print((f"Time {t:.3f} QP problem could not be solved! Error code:", sol.status))
            return np.zeros(self.robot.nv)

        return self.formulation.getActuatorForces(sol)

    @override
    def set_target(self, target_pose: pin.SE3 = None, target_q: np.array = None, target_dq: np.array = None):
        """Set target references for both end-effector and posture tasks.

        Updates the reference trajectories for both the end-effector Cartesian
        position/orientation and the joint space posture.

        Args:
            target_pose (pin.SE3, optional): Target end-effector pose in SE(3)
            target_q (np.array, optional): Target joint positions
            target_dq (np.array, optional): Target joint velocities

        Note:
            The targets can be set independently - passing None for any argument
            will keep the current reference for that task.
        """
        self.sample_end_effector.value(target_pose)
        self.eeTask.setReference(self.sample_end_effector)
        self.sample_posture.value(target_q)
        self.sample_posture.derivative(target_dq)
        self.posture_task.setReference(self.sample_posture)

    @override
    def initialize_tsid(self):
        """Initialize the TSID controller and its components.

        Sets up:
        - The inverse dynamics formulation
        - Posture task for joint space control
        - End-effector task for Cartesian space control
        - Actuation bounds for torque limits
        - Joint bounds for velocity limits
        - Reference trajectories for both tasks
        - The HQP solver

        The initialization creates all necessary TSID tasks with their respective
        gains, weights, and constraints as specified in the configuration.
        """
        robot = self.robot
        self.model = model = robot.model()

        q = self.conf.q0
        dq = np.zeros(robot.nv)

        assert model.existFrame(self.conf.end_effector_frame), (
            f"End effector frame {self.conf.end_effector_frame} not found."
        )

        self.formulation = tsid.InverseDynamicsFormulationAccForce("tsid", robot, False)
        self.formulation.computeProblemData(0.0, q, dq)

        self.posture_task = tsid.TaskJointPosture("task-posture", robot)
        self.posture_task.setKp(self.conf.Kp_posture)
        self.posture_task.setKd(self.conf.Kd_posture)
        if self.conf.posture_task_mask is not None:
            self.posture_task.setMask(self.conf.posture_task_mask)
        self.formulation.addMotionTask(self.posture_task, self.conf.w_posture, 1, 0.0)

        self.eeTask = tsid.TaskSE3Equality("task-ee", self.robot, self.conf.end_effector_frame)
        self.eeTask.setKp(self.conf.Kp_ee)
        self.eeTask.setKd(self.conf.Kd_ee)
        if self.conf.ee_task_mask is not None:
            self.eeTask.setMask(self.conf.ee_task_mask)
        self.eeTask.useLocalFrame(self.conf.ee_task_local_frame)

        self.trajector_end_effector = tsid.TrajectorySE3Constant(
            "traj-ee", self.robot.framePosition(self.formulation.data(), self.conf.enf_effector_frame_id(model))
        )
        self.formulation.addMotionTask(self.eeTask, self.conf.w_ee, 1, 0.0)

        self.tau_max = self.conf.tau_max_scaling * model.effortLimit
        self.tau_min = -self.tau_max
        self.actuation_bounds_task = tsid.TaskActuationBounds("task-actuation-bounds", robot)
        self.actuation_bounds_task.setBounds(self.tau_min, self.tau_max)
        if self.conf.w_torque_bounds > 0.0:
            self.formulation.addActuationTask(self.actuation_bounds_task, self.conf.w_torque_bounds, 0, 0.0)

        self.joint_bounds_task = tsid.TaskJointBounds("task-joint-bounds", robot, self.conf.ctrl_dt)
        self.v_max = self.conf.v_max_scaling * model.velocityLimit
        self.v_min = -self.v_max
        self.joint_bounds_task.setVelocityBounds(self.v_min, self.v_max)
        if self.conf.w_joint_bounds > 0.0:
            self.formulation.addMotionTask(self.joint_bounds_task, self.conf.w_joint_bounds, 0, 0.0)

        self.trajectory_posture = tsid.TrajectoryEuclidianConstant("traj_joint", q)
        self.posture_task.setReference(self.trajectory_posture.computeNext())

        self.solver = tsid.SolverHQuadProgFast("qp solver")
        self.solver.resize(self.formulation.nVar, self.formulation.nEq, self.formulation.nIn)

        self.sample_end_effector = self.trajector_end_effector.computeNext()
        self.sample_posture = self.trajectory_posture.computeNext()
