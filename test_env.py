from mujid.env.env import MujidEnv
import mujoco as mj
import matplotlib.pyplot as plt
import gymnasium
import numpy as np

env = MujidEnv(config={"initial_keyframe": 2, "live_view": True, "lego_shift_range": (-0.002, 0.002), "n_cameras": 2})
obs, info = env.reset()

img_1 = obs["observation.images.wrist_camera_1"]
img_2 = obs["observation.images.wrist_camera_2"]
plt.imshow(img_1)
plt.axis("off")
plt.savefig("wrist_camera_1_view_env_reset.png", bbox_inches="tight", pad_inches=0)
plt.imshow(img_2)
plt.axis("off")
plt.savefig("wrist_camera_2_view_env_reset.png", bbox_inches="tight", pad_inches=0)
# print(obs["observation.state.target"].translation)
print(obs["observation.state.cartesian"][:3])
print("Force torque sensor reading:", obs["observation.state.sensors_bota_ft_sensor"])

action = np.zeros(7)
action[3] = 1.0  # w

for _ in range(10):
    # Real, Target, Joint States, Velocities, Torques
    print(f"Real: {obs['observation.state.cartesian'][:3]}")
    print(f"Joint Positions: {obs['observation.state.joint_positions']}")
    print(f"Joint Velocities: {obs['observation.state.joint_velocities']}")
    print(f"Joint Torques: {obs['observation.state.joint_torques']}")
    # print(f"Target: {obs['observation.state.target'].translation}")
    print("Force torque sensor reading:", obs["observation.state.sensors_bota_ft_sensor"])
    obs, rew, terminated, truncated, info = env.step(action)

input()
img_1 = obs["observation.images.wrist_camera_1"]
plt.imshow(img_1)
plt.axis("off")
plt.savefig("wrist_camera_1_view_env_step.png", bbox_inches="tight", pad_inches=0)

img_2 = obs["observation.images.wrist_camera_2"]
plt.imshow(img_2)
plt.axis("off")
plt.savefig("wrist_camera_2_view_env_step.png", bbox_inches="tight", pad_inches=0)
