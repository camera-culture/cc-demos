#!/usr/bin/env python3
# view_obj_mtl.py  —  minimal OBJ + MTL viewer with PyVista

import argparse
import pyvista as pv


def main() -> None:
    ap = argparse.ArgumentParser(description="View an OBJ with its MTL")
    ap.add_argument("obj", help="Path to the .obj file")
    ap.add_argument("-m", "--mtl", help="Explicit .mtl path if needed")
    args = ap.parse_args()

    pl = pv.Plotter(window_size=(1024, 768))
    pl.import_obj(args.obj)          # loads geometry + materials
    pl.add_axes()
    pl.show()


if __name__ == "__main__":
    # from pathlib import Path
    # import pyvista as pv
    # pl = pv.Plotter()
    # pl.import_gltf(Path("/Users/aaronyoung/Downloads/cyclist.gltf").absolute())
    # pl.show()
    # from pathlib import Path
    # import trimesh
    # import pyrender

    # scene = pyrender.Scene()
    # tm_scene = trimesh.load(Path("data/car.obj").absolute())

    # for name, mesh in tm_scene.geometry.items():
    #     scene.add(pyrender.Mesh.from_trimesh(mesh))

    # pyrender.Viewer(scene, use_raymond_lighting=True)

    import moderngl_window as mglw

    class OBJViewer(mglw.WindowConfig):
        gl_version = (3, 3)
        title = "OBJ Viewer"
        window_size = (800, 600)
        resource_dir = "."

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.scene = self.load_scene("data/car.obj")
            self.camera_enabled = True

        def on_render(self, time: float, frame_time: float):
            self.ctx.clear(0.1, 0.1, 0.1)
            self.scene.draw(projection=self.camera.projection.matrix, 
                            camera_matrix=self.camera.matrix)

    mglw.run_window_config(OBJViewer)

