import sys
from pathlib import Path

import numpy as np
import trimesh
import vispy
from vispy import app, scene
from vispy.visuals import transforms
from vispy.geometry import MeshData
from vispy.io import read_mesh

try:
    from PyQt6 import QtWidgets

    app.use_app("pyqt6")
except ImportError:  # fallback
    from PyQt5 import QtWidgets

    app.use_app("pyqt5")

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


class CVPR25FullDashboard(Dashboard[CVPR25FullDashboardConfig]):
    _OBJ_Z = 5
    _GT_Z = 0.05
    _WALL_COLOR = (0, 0, 0, 1)
    _WALL_WIDTH = 3
    _WALL_H = 20

    # ---------- setup ----------
    def setup(self):
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        self.win = QtWidgets.QMainWindow()
        self.win.setWindowTitle("CVPR 2025 Full Dashboard")

        self.canvas = scene.SceneCanvas(keys="interactive", bgcolor="white", show=False)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(parent=self.view.scene)

        self.win.setCentralWidget(self.canvas.native)
        self._build_grid()
        # self._add_walls()
        self._load_static_scene(Path("data") / "scene.obj")
        self._load_static_scene(Path("data") / "road.obj", flip_faces=True, alpha=0.25)
        self._load_static_scene(
            Path("data") / "car.obj",
            alpha=1.0,
            color=(0.05, 0.15, 0.75),
        )
        self._load_object_mesh()
        self._create_gt_scatter()
        self._create_path_visualization()
        self._configure_camera()

        @self.canvas.events.key_press.connect
        def on_key_press(event):
            if event.key == " ":
                cam = self.view.camera
                print(
                    f"TurntableCamera(center={cam.center}, distance={cam.distance}, "
                    f"elevation={cam.elevation}, azimuth={cam.azimuth})"
                )
            if event.key == "escape":
                self.close()

        self.win.showFullScreen()
        self.canvas.show()

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
        color = (0.7, 0.7, 0.7, 1.0)
        for x in xs:
            scene.visuals.Line(
                pos=np.array(
                    [[x, self.config.y_range[0], 0], [x, self.config.y_range[1], 0]]
                ),
                color=color,
                width=1,
                parent=self.view.scene,
                method="gl",
            )
        for y in ys:
            scene.visuals.Line(
                pos=np.array(
                    [[self.config.x_range[0], y, 0], [self.config.x_range[1], y, 0]]
                ),
                color=color,
                width=1,
                parent=self.view.scene,
                method="gl",
            )

    def _add_wall_segment(
        self, p1: tuple[float, float], p2: tuple[float, float]
    ) -> None:
        x1, y1 = p1
        x2, y2 = p2
        z0, z1 = 0, self._WALL_H
        for x, y in [(x1, y1), (x2, y2)]:
            scene.visuals.Line(
                pos=np.array([[x, y, z0], [x, y, z1]]),
                color=self._WALL_COLOR,
                width=self._WALL_WIDTH,
                parent=self.view.scene,
                method="gl",
            )
        for z_pair in [(z1, z1), (z0, z0)]:
            scene.visuals.Line(
                pos=np.array([[x1, y1, z_pair[0]], [x2, y2, z_pair[1]]]),
                color=self._WALL_COLOR,
                width=self._WALL_WIDTH,
                parent=self.view.scene,
                method="gl",
            )

    def _add_walls(self) -> None:
        self._add_wall_segment((32, -10), (32, -30))
        self._add_wall_segment((-10, 0), (-30, 0))

    def _load_static_scene(
        self,
        model_path: Path,
        alpha=0.75,
        color=(0.8, 0.8, 0.8),
        flip_faces=False,
    ):
        mesh = trimesh.load(model_path, force="mesh")
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        )
        mesh.apply_translation([-2.5, -8.0, 0])
        mesh.apply_scale(7.5)
        if flip_faces:
            mesh.faces = mesh.faces[:, ::-1]
        self.scene_mesh = scene.visuals.Mesh(
            vertices=mesh.vertices,
            faces=mesh.faces,
            color=(*color, alpha),
            shading="smooth",
            parent=self.view.scene,
        )
        self.scene_mesh.set_gl_state(
            blend=True, depth_test=True, cull_face="back", 
        )
        self.scene_mesh.order = 1

    def _load_object_mesh(self):
        mesh = trimesh.load(self.config.obj_path, force="mesh")
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(
                angle=-np.pi / 2, direction=[1, 0, 0], point=[0, 0, 0]
            )
        )
        mesh.apply_scale(1.1)
        self.pred_mesh = scene.visuals.Mesh(
            vertices=mesh.vertices,
            faces=mesh.faces,
            color=(1, 0, 0, 1),
            shading="smooth",
            parent=self.view.scene,
        )

    def _create_gt_scatter(self):
        self.gt_dots = scene.visuals.Markers(parent=self.view.scene)
        self.gt_dots.set_data(
            pos=np.zeros((1, 3)), size=0, face_color=(0, 1, 0, 1)
        )

    def _create_path_visualization(self):
        # Create a path visualization using a line
        self.path_line = scene.visuals.Line(
            pos=np.zeros((1, 3)),
            color=(0, 0, 1, 1),
            width=100,
            parent=self.view.scene,
            method="gl",
            antialias=True
        )

    def _configure_camera(self):
        cam = self.view.camera
        cam.center = (-10, -20, 0)
        cam.distance = 85
        cam.elevation = 30
        cam.azimuth = -60

    # ---------- runtime ----------
    def update(
        self,
        frame: int,
        positions: list[tuple[float, float]] = [],
        path: np.ndarray | None = None,
        gt_positions: list[tuple[float, float]] = [],
        **_,
    ):
        if positions:
            x, y = positions[0]
            T = transforms.MatrixTransform()
            T.scale((0.05, 0.05, 0.05))
            T.rotate(-90, (0, 1, 0))
            T.translate((x, y, self._OBJ_Z))
            self.pred_mesh.transform = T

        if path is not None:
            self.path_line.set_data(pos=path)

            centroid = np.mean(path, axis=0)
            self.path_line.transform = transforms.MatrixTransform()
            self.path_line.transform.translate(-centroid)
            self.path_line.transform.rotate(180, (0, 0, 1))
            self.path_line.transform.translate(centroid)

        if gt_positions:
            pos3d = np.c_[
                np.array(gt_positions), np.full(len(gt_positions), self._GT_Z)
            ]
            self.gt_dots.set_data(
                pos=pos3d, size=self.config.dot_size, face_color=(0, 1, 0, 1)
            )

        self.canvas.update()
        self.app.processEvents()

    def run(self):
        self.app.exec()

    # ---------- housekeeping ----------
    @property
    def is_okay(self) -> bool:
        return not self.win.isHidden()

    def close(self):
        if self.win:
            self.win.close()
            self.win = None
        if self.app:
            self.app.quit()
            self.app = None


if __name__ == "__main__":
    dash = CVPR25FullDashboard(
        CVPR25FullDashboardConfig(x_range=(0, 32), y_range=(0, 32))
    )
    dash.setup()

    position = (16.0, 16.0)

    # Create a simple path for demonstration
    path = np.array([[0, 0, 0], [10, 10, 0], [20, 5, 0], [30, 15, 0]])

    try:
        while dash.is_okay:
            dash.update(frame=0, positions=[position], path=path)
    except KeyboardInterrupt:
        print("Exiting dashboard...")
        dash.close()