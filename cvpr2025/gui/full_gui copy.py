import sys
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import trimesh
from pyqtgraph.Qt import QtWidgets

from cc_hardware.tools.dashboard import Dashboard, DashboardConfig
from cc_hardware.utils import config_wrapper


@config_wrapper
class CVPR25FullDashboardConfig(DashboardConfig):
    x_range: tuple[float, float] = (-35.0, 35.0)
    y_range: tuple[float, float] = (-35.0, 35.0)
    grid_step: float = 1.0
    dot_size: int = 15
    obj_path: Path = Path("data/ped.obj")


class CVPR25FullDashboard(Dashboard[CVPR25FullDashboardConfig]):
    _OBJ_Z = 5
    _GT_Z = 0.05
    _WALL_COLOR = (0, 0, 0, 1)
    _WALL_WIDTH = 3
    _WALL_H = 20  # height

    # ---------- setup ----------
    def setup(self):
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        self.win = QtWidgets.QMainWindow()
        self.win.setWindowTitle("CVPR 2025 Full Dashboard")
        self.view = gl.GLViewWidget()
        self.win.setCentralWidget(self.view)
        self.view.setBackgroundColor("w")

        self._build_grid()
        self._add_walls()
        self._load_object_mesh()
        self._create_gt_scatter()
        self._configure_camera()

        self.win.showFullScreen()

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
        color = (0.7, 0.7, 0.7, 1)
        for x in xs:
            self.view.addItem(
                gl.GLLinePlotItem(
                    pos=np.array([[x, self.config.y_range[0], 0], [x, self.config.y_range[1], 0]]),
                    color=color,
                    width=1,
                    antialias=True,
                )
            )
        for y in ys:
            self.view.addItem(
                gl.GLLinePlotItem(
                    pos=np.array([[self.config.x_range[0], y, 0], [self.config.x_range[1], y, 0]]),
                    color=color,
                    width=1,
                    antialias=True,
                )
            )

    def _add_wall_segment(self, p1: tuple[float, float], p2: tuple[float, float]) -> None:
        """Add a rectangular wall (line prism) between p1 and p2, height _WALL_H."""
        x1, y1 = p1
        x2, y2 = p2
        z0, z1 = 0, self._WALL_H

        # vertical edges
        for (x, y) in [(x1, y1), (x2, y2)]:
            self.view.addItem(
                gl.GLLinePlotItem(
                    pos=np.array([[x, y, z0], [x, y, z1]]),
                    color=self._WALL_COLOR,
                    width=self._WALL_WIDTH,
                    antialias=True,
                )
            )
        # top edge
        self.view.addItem(
            gl.GLLinePlotItem(
                pos=np.array([[x1, y1, z1], [x2, y2, z1]]),
                color=self._WALL_COLOR,
                width=self._WALL_WIDTH,
                antialias=True,
            )
        )
        # bottom edge (for completeness, thicker than grid)
        self.view.addItem(
            gl.GLLinePlotItem(
                pos=np.array([[x1, y1, z0], [x2, y2, z0]]),
                color=self._WALL_COLOR,
                width=self._WALL_WIDTH,
                antialias=True,
            )
        )

    def _add_walls(self) -> None:
        # wall 1: x = 32, y -10 → -30
        self._add_wall_segment((32, -10), (32, -30))
        # wall 2: y = 0, x -10 → -30
        self._add_wall_segment((-10, 0), (-30, 0))

    def _load_object_mesh(self):
        mesh = trimesh.load(self.config.obj_path, force="mesh")
        mesh.apply_scale(0.05)
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(
                angle=-np.pi / 2, direction=[0, 1, 0], point=[0, 0, 0]
            )
        )
        meshdata = gl.MeshData(vertexes=mesh.vertices, faces=mesh.faces)
        self.pred_mesh = gl.GLMeshItem(
            meshdata=meshdata,
            smooth=True,
            drawFaces=True,
            color=(1, 0, 0, 1),
            shader="shaded",
        )
        self.pred_mesh.setGLOptions("opaque")
        self.view.addItem(self.pred_mesh)

    def _create_gt_scatter(self):
        self.gt_dots = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3)),
            size=self.config.dot_size,
            color=(0, 1, 0, 1),
            pxMode=True,
        )
        self.gt_dots.setGLOptions("opaque")
        self.view.addItem(self.gt_dots)

    def _configure_camera(self):
        cx = sum(self.config.x_range) / 2
        cy = sum(self.config.y_range) / 2
        span = max(
            self.config.x_range[1] - self.config.x_range[0],
            self.config.y_range[1] - self.config.y_range[0],
        )
        self.view.opts["center"] = pg.Vector(cx, cy, self._WALL_H / 2)
        self.view.setCameraPosition(distance=span * 1.8, elevation=30, azimuth=225)

    # ---------- runtime ----------
    def update(
        self,
        frame: int,
        positions: list[tuple[float, float]] = [],
        gt_positions: list[tuple[float, float]] = [],
        **_,
    ):
        if positions:
            x, y = positions[0]
            self.pred_mesh.resetTransform()
            self.pred_mesh.translate(x, y, self._OBJ_Z)

        if gt_positions:
            self.gt_dots.setData(
                pos=np.c_[
                    np.array(gt_positions),
                    np.full(len(gt_positions), self._GT_Z),
                ]
            )

        self.app.processEvents()

    def run(self):
        self.app.exec()

    # ---------- housekeeping ----------
    @property
    def is_okay(self) -> bool:
        return not self.win.isHidden()

    def close(self):
        QtWidgets.QApplication.quit()
        if self.win:
            self.win.close()
            self.win = None
        if self.app:
            self.app.quit()
            self.app = None


if __name__ == "__main__":
    dash = CVPR25FullDashboard(CVPR25FullDashboardConfig(x_range=(0, 32), y_range=(0, 32)))
    dash.setup()
    while dash.is_okay:
        dash.update(frame=0, positions=[(0.0, 0.0)], gt_positions=[(2.0, 2.0)])
