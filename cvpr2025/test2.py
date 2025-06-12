import numpy as np
import trimesh
import pyrender
from vispy import app, scene
from vispy.visuals import ImageVisual
from vispy.visuals.transforms import STTransform

# Set up pyrender scene
trimesh_scene = trimesh.load('data/scene.obj', force='scene')
scene_pyr = pyrender.Scene()
for geom in trimesh_scene.geometry.values():
    mesh = pyrender.Mesh.from_trimesh(geom, smooth=True)
    scene_pyr.add(mesh)
light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
scene_pyr.add(light, pose=np.eye(4))

renderer = pyrender.OffscreenRenderer(640, 480)

# VisPy setup
canvas = scene.SceneCanvas(keys='interactive', bgcolor='black', size=(640, 480), show=True)
view = canvas.central_widget.add_view()
view.camera = scene.cameras.PanZoomCamera(aspect=1)
image = ImageVisual(np.zeros((480, 640, 3), dtype=np.uint8))
image.transform = STTransform(scale=(1, 1), translate=(0, 0))
view.add(image)

# State
angle = 0.0

def update_scene():
    global angle
    # Rotate scene
    for node in list(scene_pyr.mesh_nodes):
        scene_pyr.remove_node(node)
    for geom in trimesh_scene.geometry.values():
        mesh = pyrender.Mesh.from_trimesh(geom.copy().apply_transform(
            trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
        ), smooth=True)
        scene_pyr.add(mesh)
    scene_pyr.add(light, pose=np.eye(4))

    # Render and update image
    color, _ = renderer.render(scene_pyr)
    image.set_data(color)

@canvas.events.key_press.connect
def on_key_press(event):
    global angle
    if event.key == 'Left':
        angle -= 0.1
    elif event.key == 'Right':
        angle += 0.1
    update_scene()

update_scene()
app.run()
