import multiprocessing as mp
from multiprocessing import shared_memory
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter
from pinocchio import pin  # type: ignore

from mujid.controllers.util import make_controller, make_throttled_logger


logger = make_throttled_logger("main", interval=5)


path_mjf = Path(__file__).parent / "mujid" / "mjcf" / "scene.xml"
path_urdf = Path(__file__).parent / "mujid" / "urdf" / "fr3_franka_hand.urdf"

ctrl_type = "cartesian_impedance"  # "tsid", "cartesian_impedance", "gravity_compensation", "inverse_dynamics"

sim_dt = 1.0 / 5000.0
max_time = 1000  # [s]

dry_friction = 0.1  # [Nm]
viscous_friction = 2.0  # [Nm/(rad/s)]

# Create shared memory for state and control exchange
STATE_SIZE = 14  # 7 for qpos + 7 for qvel
CONTROL_SIZE = 7  # Number of actuators
MOCAP_SIZE = 7  # 3 for pos + 4 for quat
FLOAT_SIZE = 8  # Size of float64 in bytes


def init_shared_memory():
    # Create shared memory blocks
    state_shm = shared_memory.SharedMemory(create=True, size=STATE_SIZE * FLOAT_SIZE)
    control_shm = shared_memory.SharedMemory(
        create=True, size=CONTROL_SIZE * FLOAT_SIZE
    )
    mocap_shm = shared_memory.SharedMemory(create=True, size=MOCAP_SIZE * FLOAT_SIZE)
    time_shm = shared_memory.SharedMemory(create=True, size=FLOAT_SIZE)

    # Initialize numpy arrays from shared memory
    state = np.ndarray((STATE_SIZE,), dtype=np.float64, buffer=state_shm.buf)
    control = np.ndarray((CONTROL_SIZE,), dtype=np.float64, buffer=control_shm.buf)
    mocap = np.ndarray((MOCAP_SIZE,), dtype=np.float64, buffer=mocap_shm.buf)
    sim_time = np.ndarray((1,), dtype=np.float64, buffer=time_shm.buf)

    return state_shm, control_shm, mocap_shm, time_shm, state, control, mocap, sim_time


def cleanup_shared_memory(state_shm, control_shm, mocap_shm, time_shm):
    state_shm.close()
    control_shm.close()
    mocap_shm.close()
    time_shm.close()
    state_shm.unlink()
    control_shm.unlink()
    mocap_shm.unlink()
    time_shm.unlink()


# logger.info(f"Simulation started with dt - {sim_dt} - and frequency - {1.0 / sim_dt}. Initial qpos: {data.qpos}")


def controller_process(shm_names, path_urdf, ctrl_type, sim_dt):
    # Reconnect to shared memory
    state_shm = shared_memory.SharedMemory(name=shm_names["state"])
    control_shm = shared_memory.SharedMemory(name=shm_names["control"])
    mocap_shm = shared_memory.SharedMemory(name=shm_names["mocap"])
    time_shm = shared_memory.SharedMemory(name=shm_names["time"])

    # Create numpy arrays from shared memory
    state = np.ndarray((STATE_SIZE,), dtype=np.float64, buffer=state_shm.buf)
    control = np.ndarray((CONTROL_SIZE,), dtype=np.float64, buffer=control_shm.buf)
    mocap = np.ndarray((MOCAP_SIZE,), dtype=np.float64, buffer=mocap_shm.buf)
    sim_time = np.ndarray((1,), dtype=np.float64, buffer=time_shm.buf)

    # Initialize controller
    ctrl, conf = make_controller(ctrl_type, str(path_urdf), sim_dt)
    rate = RateLimiter(frequency=1000.0, warn=True)

    logger.info("Controller process started")

    while True:
        # Extract current state
        qpos = state[:7]
        qvel = state[7:]
        current_time = sim_time[0]

        # Extract mocap data
        mocap_pos = mocap[:3]
        mocap_quat = mocap[3:]

        # Set the target pose
        target_pose = pin.SE3(
            pin.Quaternion(
                x=mocap_quat[1], y=mocap_quat[2], z=mocap_quat[3], w=mocap_quat[0]
            ),
            mocap_pos,
        )
        ctrl.set_target(target_pose, target_q=conf.q0, target_dq=np.zeros(7))

        # Compute control
        desired_torques = ctrl.update(current_time, qpos, qvel)

        # Update shared control array
        control[:] = desired_torques

        rate.sleep()


def run():
    frequency_rl = 15
    frequency_controller = 900
    frequency_simulation = 4500
    n_controller_steps_per_rl_step = frequency_controller // frequency_rl
    n_simulation_steps_per_controller_step = (
        frequency_simulation // frequency_controller
    )
    viz_update_rate = 50  # Update viewer every N simulation steps

    state = np.zeros((STATE_SIZE,), dtype=np.float64)
    control = np.zeros((CONTROL_SIZE,), dtype=np.float64)
    mocap = np.zeros((MOCAP_SIZE,), dtype=np.float64)
    sim_time = np.zeros((1,), dtype=np.float64)

    # Initialize MuJoCo simulation
    spec = mujoco.MjSpec.from_file(str(path_mjf))
    spec.option.timestep = 1.0 / frequency_simulation
    model = spec.compile()

    # Set friction values for all robot joints
    for i in range(7):
        joint_id = model.joint(f"fr3_joint{i + 1}").id
        model.dof_damping[joint_id] = viscous_friction  # Viscous friction coefficient
        model.dof_frictionloss[joint_id] = dry_friction  # Dry/Coulomb friction

    data = mujoco.MjData(model)

    # Reset to initial state
    keyframe_id = 1
    mujoco.mj_resetDataKeyframe(model, data, keyframe_id)

    # Initialize controller
    ctrl, conf = make_controller(ctrl_type, str(path_urdf), sim_dt)

    logger.info("Controller process started")
    step_counter = 0
    rate = RateLimiter(frequency=1.0 / frequency_simulation, warn=False)

    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = False
        while viewer.is_running():
            for i in range(n_controller_steps_per_rl_step):
                for j in range(n_simulation_steps_per_controller_step):
                    # Update simulation with control inputs
                    data.ctrl = control

                    # Step simulation
                    mujoco.mj_step(model, data)

                    # Update shared state
                    state[:7] = data.qpos
                    if step_counter % 10000 == 0:
                        print(data.qpos[:3])
                    state[7:] = data.qvel
                    sim_time[0] = data.time

                    # Update shared mocap state
                    mocap[:3] = data.mocap_pos[0]
                    mocap[3:] = data.mocap_quat[0]
                    if step_counter % viz_update_rate == 0:
                        viewer.sync()
                    step_counter += 1
                    # if j < n_simulation_steps_per_controller_step - 1:
                    #     rate.sleep()

                # Extract current state
                qpos = state[:7]
                qvel = state[7:]
                current_time = sim_time[0]

                # Extract mocap data
                mocap_pos = mocap[:3]
                mocap_quat = mocap[3:]

                # Set the target pose
                target_pose = pin.SE3(
                    pin.Quaternion(
                        x=mocap_quat[1],
                        y=mocap_quat[2],
                        z=mocap_quat[3],
                        w=mocap_quat[0],
                    ),
                    mocap_pos,
                )
                ctrl.set_target(target_pose, target_q=conf.q0, target_dq=np.zeros(7))

                # Compute control
                desired_torques = ctrl.update(current_time, qpos, qvel)

                # Update shared control array
                control[:] = desired_torques
                # rate.sleep()


def simulation_process(
    shm_names, path_mjf, sim_dt, max_time, viz_update_rate: int = 100
):
    # Initialize MuJoCo simulation
    spec = mujoco.MjSpec.from_file(str(path_mjf))
    spec.option.timestep = sim_dt
    model = spec.compile()

    # Set friction values for all robot joints
    for i in range(7):
        joint_id = model.joint(f"fr3_joint{i + 1}").id
        model.dof_damping[joint_id] = viscous_friction  # Viscous friction coefficient
        model.dof_frictionloss[joint_id] = dry_friction  # Dry/Coulomb friction

    data = mujoco.MjData(model)

    # Reset to initial state
    keyframe_id = 1
    mujoco.mj_resetDataKeyframe(model, data, keyframe_id)

    # Reconnect to shared memory
    state_shm = shared_memory.SharedMemory(name=shm_names["state"])
    control_shm = shared_memory.SharedMemory(name=shm_names["control"])
    mocap_shm = shared_memory.SharedMemory(name=shm_names["mocap"])
    time_shm = shared_memory.SharedMemory(name=shm_names["time"])

    # Create numpy arrays from shared memory
    state = np.ndarray((STATE_SIZE,), dtype=np.float64, buffer=state_shm.buf)
    control = np.ndarray((CONTROL_SIZE,), dtype=np.float64, buffer=control_shm.buf)
    mocap = np.ndarray((MOCAP_SIZE,), dtype=np.float64, buffer=mocap_shm.buf)
    sim_time = np.ndarray((1,), dtype=np.float64, buffer=time_shm.buf)

    rate = RateLimiter(frequency=1.0 / sim_dt, warn=False)

    logger.info("Simulation process started")

    step_counter = 0
    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        start = time.time()

        while viewer.is_running() and time.time() - start < max_time:
            # Update simulation with control inputs
            data.ctrl = control

            # Step simulation
            mujoco.mj_step(model, data)

            # Update shared state
            state[:7] = data.qpos
            state[7:] = data.qvel
            sim_time[0] = data.time

            # Update shared mocap state
            mocap[:3] = data.mocap_pos[0]
            mocap[3:] = data.mocap_quat[0]

            if step_counter % viz_update_rate == 0:
                viewer.sync()

            logger.info(
                f"Current sim-time : {data.time} - Current time from start: {time.time() - start}"
            )
            rate.sleep()

            step_counter += 1


if __name__ == "__main__":
    run()
    sys.exit(0)
    try:
        # Initialize shared memory
        state_shm, control_shm, mocap_shm, time_shm, state, control, mocap, sim_time = (
            init_shared_memory()
        )

        # Create dictionary of shared memory names
        shm_names = {
            "state": state_shm.name,
            "control": control_shm.name,
            "mocap": mocap_shm.name,
            "time": time_shm.name,
        }

        # Create processes
        ctrl_process = mp.Process(
            target=controller_process, args=(shm_names, path_urdf, ctrl_type, sim_dt)
        )

        sim_process = mp.Process(
            target=simulation_process, args=(shm_names, path_mjf, sim_dt, max_time)
        )

        # Start processes
        ctrl_process.start()
        sim_process.start()

        # Wait for simulation to finish
        sim_process.join()

        # Terminate controller process
        ctrl_process.terminate()
        ctrl_process.join()

    finally:
        # Cleanup shared memory
        cleanup_shared_memory(state_shm, control_shm, mocap_shm, time_shm)
