#!/usr/bin/env python3
import sys
from pathlib import Path

import numpy as np
import trimesh
from trimesh.transformations import (
    rotation_matrix,
    scale_matrix,
    translation_matrix,
)
import pyrender
from pyrender import MetallicRoughnessMaterial as Mat
from cc_hardware.tools.dashboard import Dashboard, DashboardConfig
from cc_hardware.utils import config_wrapper


@config_wrapper
class CVPR25FullDashboardConfig(DashboardConfig):
    x_range: tuple[float, float] = (-35.0, 35.0)
    y_range: tuple[float, float] = (-35.0, 35.0)
    grid_step: float = 1.0
    dot_size: int = 15
    obj_path: Path = Path("data/ped.obj")
    scene_path: Path = Path("data/scene.obj")


def _cam_pose(center, dist, elev, azim):
    az, el = np.radians([azim, elev])
    eye = np.array(
        [
            center[0] + dist * np.cos(el) * np.cos(az),
            center[1] + dist * np.cos(el) * np.sin(az),
            center[2] + dist * np.sin(el),
        ]
    )
    f = center - eye
    f /= np.linalg.norm(f)
    up = np.array([0, 0, 1], dtype=float)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    T = np.eye(4)
    T[:3, 0] = s
    T[:3, 1] = u
    T[:3, 2] = -f
    T[:3, 3] = eye
    return T


class CVPR25FullDashboard(Dashboard[CVPR25FullDashboardConfig]):
    _OBJ_Z = 0
    _GT_Z = 0.05
    _WALL_COLOR = (0, 0, 0, 1)
    _WALL_WIDTH = 0.1
    _WALL_H = 20

    # ---------- setup ----------
    def setup(self):
        self.scene = pyrender.Scene(
            bg_color=[1, 1, 1, 1], ambient_light=[0.6, 0.6, 0.6]
        )
        # self._build_grid()
        # self._add_walls()
        self._load_static_scene(Path("data") / "scene.obj")
        self._load_object_mesh()
        self.gt_node = None
        self.path_node = None
        cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        self.cam_node = self.scene.add(
            cam,
            pose=_cam_pose(center=(-10, -20, 0), dist=85, elev=30, azim=-60),
        )
        self.scene.add(
            pyrender.DirectionalLight(color=np.ones(3), intensity=3.0),
        )
        self.viewer = pyrender.Viewer(
            self.scene,
            use_raymond_lighting=True,
            run_in_thread=True,
            viewport_size=(1280, 800),
            background_color=[.0, .0, .0, 1.0],
        )

    # ---------- helpers ----------
    def _build_grid(self):
        xs = np.arange(
            self.config.x_range[0],
            self.config.x_range[1] + self.config.grid_step,
            self.config.grid_step,
        )
        ys = np.arange(
            self.config.y_range[0],
            self.config.y_range[1] + self.config.grid_step,
            self.config.grid_step,
        )
        grid_mat = Mat(baseColorFactor=(0.7, 0.7, 0.7, 1))
        for x in xs:
            path = trimesh.load_path(
                np.array(
                    [[x, self.config.y_range[0], 0], [x, self.config.y_range[1], 0]]
                )
            )
            self.scene.add(pyrender.Mesh.from_trimesh(path, material=grid_mat))
        for y in ys:
            path = trimesh.load_path(
                np.array(
                    [[self.config.x_range[0], y, 0], [self.config.x_range[1], y, 0]]
                )
            )
            self.scene.add(pyrender.Mesh.from_trimesh(path, material=grid_mat))

    def _load_static_scene(self, model_path):
        scene = trimesh.load(model_path, force="scene")
        transform = (
            trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
            @ trimesh.transformations.scale_matrix(7.5)
            @ trimesh.transformations.translation_matrix([-2.5, -8.0, 0])
        )

        for name, geom in scene.geometry.items():
            mesh = pyrender.Mesh.from_trimesh(geom, smooth=True)
            self.scene.add(mesh, pose=transform)

    def _load_object_mesh(self):
        mesh = trimesh.load(self.config.obj_path, force="mesh")
        # translate down to match the ground plane
        mesh.apply_translation([0, 0, self._OBJ_Z])
        mesh.apply_transform(rotation_matrix(-np.pi / 2, [0, 1, 0]))
        mesh.apply_scale(0.25)
        mat = Mat(baseColorFactor=(1, 0, 0, 1))
        self.pred_mesh = pyrender.Mesh.from_trimesh(mesh, material=mat, smooth=True)
        self.pred_node = self.scene.add(self.pred_mesh)

    # ---------- runtime ----------
    def update(
        self,
        frame: int,
        positions: list[tuple[float, float]] = [],
        path: np.ndarray | None = None,
        gt_positions: list[tuple[float, float]] = [],
        **_,
    ):
        with self.viewer.render_lock:
            if positions:
                x, y = positions[0]
                T = (
                    scale_matrix(0.05)
                    @ rotation_matrix(-np.pi / 2, [0, 1, 0])
                    @ translation_matrix([x, y, self._OBJ_Z])
                )
                self.pred_node.matrix = T

            # if path is not None:
            #     if self.path_node:
            #         self.scene.remove_node(self.path_node)
            #     path_trimesh = trimesh.load_path(path)
            #     mat = Mat(baseColorFactor=(0, 0, 1, 1))
            #     self.path_node = self.scene.add(
            #         pyrender.Mesh.from_trimesh(path_trimesh, material=mat)
            #     )

            if gt_positions:
                if self.gt_node:
                    self.scene.remove_node(self.gt_node)
                spheres = []
                for gx, gy in gt_positions:
                    s = trimesh.creation.icosphere(
                        subdivisions=2, radius=self.config.dot_size * 0.01
                    )
                    s.apply_translation([gx, gy, self._GT_Z])
                    spheres.append(s)
                if spheres:
                    gt_mesh = trimesh.util.concatenate(spheres)
                    mat = Mat(baseColorFactor=(0, 1, 0, 1))
                    self.gt_node = self.scene.add(
                        pyrender.Mesh.from_trimesh(gt_mesh, material=mat)
                    )

    def run(self):
        raise NotImplementedError(
            "This dashboard is intended to be run as a script, not as a component."
        )

    # ---------- housekeeping ----------
    @property
    def is_okay(self) -> bool:
        return self.viewer.is_active

    def close(self):
        if self.viewer:
            self.viewer.close_external()
            self.viewer = None


if __name__ == "__main__":
    dash = CVPR25FullDashboard(
        CVPR25FullDashboardConfig(x_range=(0, 32), y_range=(0, 32))
    )
    dash.setup()

    position = (16.0, 16.0)
    path = np.array([[0, 0, 0], [10, 10, 0], [20, 5, 0], [30, 15, 0]])

    try:
        while dash.is_okay:
            dash.update(frame=0, positions=[position], path=path)
    except KeyboardInterrupt:
        dash.close()
