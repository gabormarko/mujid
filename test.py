import mujoco as mj
import matplotlib.pyplot as plt


def render(model, data=None, height=300, camera=0):
    if data is None:
        data = mj.MjData(model)
    with mj.Renderer(model, 480, 640) as renderer:
        mj.mj_forward(model, data)
        renderer.update_scene(data, camera)
        img = renderer.render()
        plt.imshow(img)
        plt.axis("off")
        plt.savefig("fr3_egocentric_view.png", bbox_inches="tight", pad_inches=0)


spec = mj.MjSpec.from_file("mujid/mjcf/fr3.xml")
spec.option.timestep = 0.001
model = spec.compile()
render(model)
