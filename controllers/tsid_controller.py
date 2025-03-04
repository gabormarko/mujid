import tsid
import numpy as np
import pinocchio as pin
from dataclasses import dataclass
from controllers.controller import Controller


@dataclass
class TSIDConfig:
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

    w_ee = 1.0
    w_posture = 1e-3
    w_torque_bounds = 1.0
    w_joint_bounds = 1.0

    kp_ee = 100.0
    kp_posture = 100.0

    posture_task_mask = np.array([1, 1, 1, 1, 1, 1, 0])
    joint_weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    tau_max_scaling = 0.4
    v_max_scaling = 0.8

    ee_frame_name = "fr3_hand_tcp"
    # ee_task_mask = np.array([1, 1, 1, 0, 0, 0])
    ee_task_mask = np.ones(6)
    ee_task_local_frame = True  # specifies whether task is formulated in local frame


class TSIDJointConfig(TSIDConfig):
    w_ee = 0.0
    kp_posture = 100.0

    joint_weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 20.0])


class TSIDController(Controller):
    def __init__(self, conf: TSIDConfig = None, dt: float = 0.001, **kwargs):
        super().__init__(**kwargs)
        self.conf = conf if conf is not None else TSIDConfig()
        self.robot = tsid.RobotWrapper(self.path_to_urdf, [], False)
        self.dt = dt
        self._n_calls = 0
        self.initialize_tsid()

    def update(self, q: np.array, dq: np.array) -> np.array:
        self._n_calls += 1
        t = self._n_calls * self.dt

        # self._update_robot_model(q, dq)
        # self.postureTask.setKp(posture_Kp)
        # self.postureTask.setKd(posture_Kd)

        HQPData = self.formulation.computeProblemData(t, q, dq)

        sol = self.solver.solve(HQPData)
        if sol.status != 0:
            print((f"Time {t:.3f} QP problem could not be solved! Error code:", sol.status))

        return self.formulation.getActuatorForces(sol)

    def set_target(self, target_pose: pin.SE3 = None, target_q: np.array = None, target_dq: np.array = None):
        self.sampleEE.value(target_pose)
        self.eeTask.setReference(self.sampleEE)
        self.samplePosture.value(target_q)
        self.samplePosture.derivative(target_dq)
        self.postureTask.setReference(self.samplePosture)

    def initialize_tsid(self):
        robot = self.robot
        self.model = model = robot.model()

        q = self.conf.q0
        dq = np.zeros(robot.nv)

        assert model.existFrame(self.conf.ee_frame_name)

        formulation = tsid.InverseDynamicsFormulationAccForce("tsid", robot, False)
        formulation.computeProblemData(0.0, q, dq)

        self.postureTask = tsid.TaskJointPosture("task-posture", robot)

        posture_Kp = self.conf.kp_posture * self.conf.joint_weights
        posture_Kd = 2.0 * np.sqrt(self.conf.kp_posture) * self.conf.joint_weights
        self.postureTask.setKp(posture_Kp)
        self.postureTask.setKd(posture_Kd)
        self.postureTask.setMask(self.conf.posture_task_mask)

        formulation.addMotionTask(self.postureTask, self.conf.w_posture, 1, 0.0)

        self.eeTask = tsid.TaskSE3Equality("task-ee", self.robot, self.conf.ee_frame_name)
        kp = self.conf.kp_ee * np.eye(6) if isinstance(self.conf.kp_ee, float) else self.conf.kp_ee
        kd = 2.0 * np.sqrt(kp)
        self.eeTask.setKp(kp)
        self.eeTask.setKd(kd)
        self.eeTask.setMask(self.conf.ee_task_mask)
        self.eeTask.useLocalFrame(self.conf.ee_task_local_frame)

        self.EE = model.getFrameId(self.conf.ee_frame_name)
        H_ee_ref = self.robot.framePosition(formulation.data(), self.EE)
        self.trajEE = tsid.TrajectorySE3Constant("traj-ee", H_ee_ref)
        formulation.addMotionTask(self.eeTask, self.conf.w_ee, 1, 0.0)

        self.tau_max = self.conf.tau_max_scaling * model.effortLimit
        self.tau_min = -self.tau_max
        actuationBoundsTask = tsid.TaskActuationBounds("task-actuation-bounds", robot)
        actuationBoundsTask.setBounds(self.tau_min, self.tau_max)
        if self.conf.w_torque_bounds > 0.0:
            formulation.addActuationTask(actuationBoundsTask, self.conf.w_torque_bounds, 0, 0.0)

        jointBoundsTask = tsid.TaskJointBounds("task-joint-bounds", robot, self.dt)
        self.v_max = self.conf.v_max_scaling * model.velocityLimit
        self.v_min = -self.v_max
        jointBoundsTask.setVelocityBounds(self.v_min, self.v_max)
        if self.conf.w_joint_bounds > 0.0:
            formulation.addMotionTask(jointBoundsTask, self.conf.w_joint_bounds, 0, 0.0)

        trajPosture = tsid.TrajectoryEuclidianConstant("traj_joint", q)
        self.postureTask.setReference(trajPosture.computeNext())

        solver = tsid.SolverHQuadProgFast("qp solver")
        solver.resize(formulation.nVar, formulation.nEq, formulation.nIn)

        self.trajPosture = trajPosture
        self.actuationBoundsTask = actuationBoundsTask
        self.jointBoundsTask = jointBoundsTask
        self.formulation = formulation
        self.solver = solver
        self.q = q
        self.dq = dq

        self.sampleEE = self.trajEE.computeNext()
        self.samplePosture = self.trajPosture.computeNext()
