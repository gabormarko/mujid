import gymnasium
import mujoco
import mujoco.viewer
import numpy as np
import pinocchio as pin

from mujid.controllers.util import make_controller


class MujidEnv(gymnasium.Env):
    path_mjf = "/home/linus/uni/master/code/mujid/mujid/mjcf/scene_no_mocap.xml"
    path_urdf = "/home/linus/uni/master/code/mujid/mujid/urdf/fr3_franka_hand.urdf"

    ctrl_type = "cartesian_impedance"  # "tsid", "cartesian_impedance", "gravity_compensation", "inverse_dynamics"

    dry_friction = 0.1  # [Nm]
    viscous_friction = 2.0  # [Nm/(rad/s)]

    # Create shared memory for state and control exchange
    STATE_SIZE = 14  # 7 for qpos + 7 for qvel
    CONTROL_SIZE = 7  # Number of actuators
    action_space = gymnasium.spaces.Box(
        low=np.array([-0.05, -0.05, -0.05, -1, -1, -1, -1]),
        high=np.array([0.05, 0.05, 0.05, 1, 1, 1, 1]),
        dtype=np.float32,
    )
    frequency_rl = 15
    frequency_controller = 500
    frequency_simulation = 3000
    sim_dt = 1.0 / frequency_simulation
    n_controller_steps_per_rl_step = frequency_controller // frequency_rl
    n_simulation_steps_per_controller_step = (
        frequency_simulation // frequency_controller
    )
    # viz_update_rate = 50  # Update viewer every N simulation steps
    camera_id = 0
    CAM_HEIGHT = 256
    CAM_WIDTH = 256
    cameras = ["wrist_cam"]
    n_init = 0

    def __init__(self, config):
        super().__init__()
        if MujidEnv.n_init > 0:
            assert False, "Env should only be created once"
        else:
            MujidEnv.n_init += 1

        self.config = config

        # Configurable lego randomization range (x-axis). Provide as (low, high).
        # Example: config['lego_shift_range'] = (-0.01, 0.01)
        self.lego_shift_range = config.get("lego_shift_range", (-0.002, 0.002))
        self.initial_position_range = config.get(
            "initial_position_range",
            (np.array([-0.0, -0.0, -0.0]), np.array([0.0, 0.0, 0.0])),
        )

        # Initialize MuJoCo simulation
        self.spec_ = mujoco.MjSpec.from_file(str(self.path_mjf))
        self.spec_.option.timestep = 1.0 / self.frequency_simulation
        self.model = self.spec_.compile()

        # Record lego body id and its original relative position so we can
        # apply small shifts at reset.
        self._gripped_lego_body_id = self.model.body("lego_2x2_hollow").id
        # Copy original model-relative body position (x,y,z)
        self._gripped_lego_body_pos0 = self.model.body_pos[
            self._gripped_lego_body_id
        ].copy()

        # Set friction values for all robot joints
        for i in range(7):
            joint_id = self.model.joint(f"fr3_joint{i + 1}").id
            self.model.dof_damping[joint_id] = (
                self.viscous_friction  # Viscous friction coefficient
            )
            self.model.dof_frictionloss[joint_id] = (
                self.dry_friction
            )  # Dry/Coulomb friction

        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(
            self.model,
            self.data,
            key=config["initial_keyframe"] if "initial_keyframe" in config else 1,
        )
        mujoco.mj_forward(self.model, self.data)

        # initialize the renderer
        self.renderer = mujoco.Renderer(self.model, self.CAM_HEIGHT, self.CAM_WIDTH)
        self.cam_buffer = np.zeros((self.CAM_HEIGHT, self.CAM_WIDTH, 3), dtype=np.uint8)

        # Initialize controller
        self.ctrl, self.conf = make_controller(
            self.ctrl_type, str(self.path_urdf), self.sim_dt
        )

        if config.get("live_view", False):
            self.viewer = mujoco.viewer.launch_passive(
                self.model, self.data, show_left_ui=False, show_right_ui=False
            )
        else:
            self.viewer = None

        self.target_pose = pin.SE3(
            pin.Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),  # type: ignore
            np.zeros(3),
        )

    def step(self, action: np.ndarray, block=False):
        # Set the target pose
        self.target_pose = pin.SE3(
            self.target_pose.rotation
            @ pin.Quaternion(
                x=action[4],
                y=action[5],
                z=action[6],
                w=action[3],
            ).matrix(),  # type: ignore
            self.target_pose.translation + action[:3],
        )
        self.ctrl.set_target(
            self.target_pose, target_q=self.conf.q0, target_dq=np.zeros(7)
        )

        for _ in range(self.n_controller_steps_per_rl_step):
            self.data.ctrl = self.ctrl.update(0.0, self.data.qpos, self.data.qvel)
            mujoco.mj_step(
                self.model, self.data, nstep=self.n_simulation_steps_per_controller_step
            )

        return self._get_obs(), 0.0, False, False, {}

    def reset(self, seed=None, options=None):
        # Optionally apply a small random x-shift to the lego body before
        # resetting the simulation state so the change is reflected in the
        # initial data.
        lego_shift_low, lego_shift_high = self.lego_shift_range
        lego_dx = float(np.random.uniform(lego_shift_low, lego_shift_high))
        # Apply shift in model-relative coordinates (x axis)
        self.model.body_pos[self._gripped_lego_body_id, 0] = (
            self._gripped_lego_body_pos0[0] + lego_dx
        )

        initial_dxyz_low, initial_dxyz_high = self.initial_position_range
        initial_dxyz = np.random.uniform(initial_dxyz_low, initial_dxyz_high)

        # Reset to initial state (home)
        mujoco.mj_resetDataKeyframe(
            self.model,
            self.data,
            key=(
                self.config["initial_keyframe"]
                if "initial_keyframe" in self.config
                else 1
            ),
        )
        mujoco.mj_forward(self.model, self.data)

        self.target_pose: pin.SE3 = pin.SE3(
            self.data.site("fr3_hand_tcp").xmat.reshape(3, 3),
            self.data.site("fr3_hand_tcp").xpos,
        )

        # Adjust initial pose to account for lego shift and randomization by running the controller
        self.step(
            np.array(
                [
                    initial_dxyz[0] - lego_dx,
                    initial_dxyz[1],
                    initial_dxyz[2],
                    1,
                    0,
                    0,
                    0,
                ]
            )
        )  # Initial pose
        for _ in range(3 * self.frequency_rl):  # warm-up steps, 3 seconds
            self.step(np.array([0.0, 0, 0, 1, 0, 0, 0]))

        return self._get_obs(), {
            "reset.grasped.position": np.array([-lego_dx, 0.0, 0.0])
        }

    def render(self, mode="human"):
        # Implement the render logic
        pass

    def close(self):
        # Implement any cleanup logic
        if self.renderer is not None:
            self.renderer.close()
        pass

    def _get_obs(self):
        return {
            "observation.images.wrist_camera": self._render(),
            "observation.state.target": np.concatenate(
                [
                    self.target_pose.translation,
                    quaternion_from_rotation_matrix(self.target_pose.rotation),
                ]
            ),  # type: ignore
            "observation.state.joint_positions": self.data.qpos.copy(),
            "observation.state.joint_velocities": self.data.qvel.copy(),
            "observation.state.joint_torques": self.data.ctrl.copy(),
            "observation.state.cartesian": self.data.site("fr3_hand_tcp").xpos,
            "observation.state.moving_brick": self.data.geom("wall_top").xpos,
        }

    def _render(self):
        self.renderer.update_scene(self.data, self.camera_id)
        if self.viewer is not None:
            self.viewer.sync()
        return self.renderer.render(out=self.cam_buffer)


def quaternion_from_rotation_matrix(mat: np.ndarray) -> np.ndarray:
    cos_theta = (np.trace(mat) - 1.0) / 2.0
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    axis = np.array(
        [
            mat[2, 1] - mat[1, 2],
            mat[0, 2] - mat[2, 0],
            mat[1, 0] - mat[0, 1],
        ]
    )
    axis_norm = np.linalg.norm(axis)
    axis = axis * ((np.sin(theta / 2.0) / axis_norm) if axis_norm > 5e-5 else 0)

    return np.array([np.cos((theta / 2.0)), axis[0], axis[1], axis[2]])
