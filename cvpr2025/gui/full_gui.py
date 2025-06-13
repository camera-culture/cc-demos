import sys
from pathlib import Path
from collections import deque
import time

import numpy as np
import trimesh
from vispy import app, scene, visuals
from vispy.visuals import transforms
from vispy.io import read_mesh  # noqa: F401

# -------------------- Qt / PyQtGraph setup --------------------
try:
    from PyQt6 import QtWidgets, QtGui, QtCore

    QT_BACKEND = "pyqt6"
except ImportError:  # fallback
    from PyQt5 import QtWidgets, QtGui, QtCore

    QT_BACKEND = "pyqt5"

app.use_app(QT_BACKEND)

import pyqtgraph as pg  # noqa: E402

# -------------------- Histogram constants --------------------
START_BIN = 0
END_BIN = 18

# -------------------- Config --------------------
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


# -------------------- Histogram widget --------------------
class HistogramWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAutoFillBackground(False)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setBackground(None)
        layout.addWidget(self.plot_widget)

        self.setFixedSize(320, 200)
        self._create_plot()

    def _create_plot(self):
        self.bins = np.arange(START_BIN, END_BIN)
        self.plot: pg.PlotItem = self.plot_widget.addPlot()
        self.bar_item = pg.BarGraphItem(
            x=self.bins + 0.5,
            height=np.zeros_like(self.bins, dtype=float),
            width=1.0,
            brush=QtGui.QColor(0, 100, 255, 160),
        )
        self.plot.addItem(self.bar_item)
        self.plot.setLabel("bottom", "Time Bin")
        self.plot.setLabel("left", "# Photons")
        self.plot.setXRange(START_BIN, END_BIN, padding=0)
        self.plot.enableAutoRange(axis="y", enable=True)

    def update(self, *, histograms: np.ndarray):
        # `histograms` can be 1-D (length 16) or N×16; average if >1-D
        if histograms.ndim > 1:
            data = histograms.mean(axis=0)
        else:
            data = histograms
        self.bar_item.setOpts(height=data)

    # rounded translucent background
    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setBrush(QtGui.QColor(255, 255, 255, 100))
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawRoundedRect(self.rect(), 6, 6)

class CameraWidget(QtWidgets.QWidget):
    def __init__(self, parent=None, size=(320, 240)):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAutoFillBackground(False)
        self.setFixedSize(*size)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QtWidgets.QLabel()
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setScaledContents(True)
        layout.addWidget(self.label)

    def update(self, *, image: np.ndarray):
        if image.ndim == 2:                       # grayscale → RGB
            image = np.stack([image] * 3, axis=-1)
        if image.dtype != np.uint8:               # ensure uint8
            image = np.clip(image, 0, 255).astype(np.uint8)
        h, w, _ = image.shape
        # crop center 2/3rds of image in width
        # image = image[:, w // 6: -w // 6, :]
        # h, w = image.shape[:2]
        image = np.ascontiguousarray(image)  # ensure contiguous memory layout
        qimg = QtGui.QImage(image.data, w, h, 3 * w,
                            QtGui.QImage.Format.Format_RGB888)
        self.label.setPixmap(QtGui.QPixmap.fromImage(qimg))

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setBrush(QtGui.QColor(255, 255, 255, 100))
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.drawRoundedRect(self.rect(), 6, 6)


# -------------------- Dashboard --------------------
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

        self._load_static_scene(Path("data") / "scene.obj")
        self._load_static_scene(Path("data") / "road.obj", flip_faces=True, alpha=0.25)
        self._load_static_scene(Path("data") / "lines.obj", alpha=0.5, color=(1, 1, 1))
        self._load_static_scene(
            Path("data") / "car.obj",
            alpha=1.0,
            color=(0.05, 0.15, 0.75),
        )
        self.stop_line = self._load_static_scene(
            Path("data") / "stop_line.obj",
            alpha=1.0,
            color=(0.8, 0.2, 0.2),
        )
        # self.stop_sign = self._load_static_scene(
        #     Path("data") / "stop_sign.obj",
        #     alpha=1.0,
        #     color=(0.8, 0.2, 0.2),
        #     hide=True,
        # )
        # self.stop_sign_text = self._load_static_scene(
        #     Path("data") / "stop_sign_text.obj",
        #     alpha=1.0,
        #     color=(1.0, 1.0, 1.0),
        #     hide=True,
        # )
        self.caution_sign = self._load_static_scene(
            Path("data") / "caution_sign.obj",
            alpha=1.0,
            color=(0.9, 0.75, 0.15),
            hide=True,
        )
        self._load_object_mesh()
        self._create_gt_scatter()
        self._create_path_visualization()
        self._configure_camera()

        # velocity history (last N frames) and arrow visual
        self._vel_hist = deque(maxlen=8)          # N = 8 frames
        # self._arrow = scene.visuals.Arrow(
        #     pos=np.zeros((2, 3)),
        #     color=self.pred_mesh.color,
        #     arrow_color=self.pred_mesh.color,
        #     arrow_size=12,
        #     arrow_type="triangle_60",
        #     width=2,
        #     parent=self.view.scene,
        # )
        # self._arrow.visible = False


        # histogram overlay
        self.histogram_widget = HistogramWidget(self.win)
        self.camera_widget = CameraWidget(self.win)

        self.win.resize(1280, 800)
        self.win.showFullScreen()
        self.canvas.show()

        self._place_histogram()
        self._place_camera()

        self.win.installEventFilter(self.win)

    # ---------- helpers ----------
    def eventFilter(self, obj, ev):
        if ev.type() == QtCore.QEvent.Type.Resize and obj is self.win:
            self._place_histogram()
            self._place_camera()
        return super().eventFilter(obj, ev)

    def _place_histogram(self):
        margin = 12
        h = self.win.size().height()
        self.histogram_widget.move(margin, h - self.histogram_widget.height() - margin)  # bottom-left

    def _place_camera(self):
        margin = 12
        self.camera_widget.move(margin, margin)        # top-left

    def _load_static_scene(
        self,
        model_path: Path,
        alpha=0.75,
        color=(0.8, 0.8, 0.8),
        flip_faces=False,
        hide=False,
    ):
        mesh = trimesh.load(model_path, force="mesh")
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        )
        mesh.apply_translation([-2.5, -8.0, 0])
        mesh.apply_scale(7.5)
        if flip_faces:
            mesh.faces = mesh.faces[:, ::-1]
        scene_mesh = scene.visuals.Mesh(
            vertices=mesh.vertices,
            faces=mesh.faces,
            color=(*color, alpha),
            shading="smooth",
            parent=self.view.scene,
        )
        scene_mesh.set_gl_state(blend=True, depth_test=True, cull_face="back")
        scene_mesh.order = 1
        scene_mesh.visible = not hide
        return scene_mesh

    def _load_object_mesh(self):
        mesh = trimesh.load(self.config.obj_path, force="mesh")
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(
                angle=np.pi / 2, direction=[1, 0, 0], point=[0, 0, 0]
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
        self.gt_dots.set_data(pos=np.zeros((1, 3)), size=0, face_color=(0, 1, 0, 1))

    def _create_path_visualization(self):
        self.path_line = scene.visuals.Line(
            pos=np.zeros((1, 3)),
            color=(0, 0, 1, 1),
            width=100,
            parent=self.view.scene,
            method="gl",
            antialias=True,
        )

    def _configure_camera(self):
        cam = self.view.camera
        cam.center = (-10, -20, 0)
        cam.distance = 85
        cam.elevation = 17.5
        cam.azimuth = -34

    # ---------- runtime ----------
    def update(
        self,
        frame: int,
        positions: list[tuple[float, float]] = [],
        path: np.ndarray | None = None,
        gt_positions: list[tuple[float, float]] = [],
        histograms: np.ndarray | None = None,
        image: np.ndarray | None = None,
        **_,
    ):
        x, y = positions[0]
        self.pred_mesh.color = (0, 1, 0, 1) if y > 16 else (1, 0, 0, 1)
        T = transforms.MatrixTransform()
        T.scale((0.05, 0.05, 0.05))
        T.rotate(-90, (0, 1, 0))
        T.translate((x, y, self._OBJ_Z))
        self.pred_mesh.transform = T

        # # --- direction-of-travel arrow ---
        # now = time.time()
        # self._vel_hist.append((now, np.array([x, y, self._OBJ_Z])))

        # if len(self._vel_hist) > 1:
        #     t0, p0 = self._vel_hist[0]
        #     tn, pn = self._vel_hist[-1]
        #     dt = tn - t0
        #     if dt > 0:
        #         v = (pn - p0) / dt              # velocity vector (units/sec)
        #         speed = np.linalg.norm(v)
        #         if speed > 1e-3:                # ignore tiny movement
        #             scale = 0.5                 # arrow length scaling factor
        #             end = pn + (v / speed) * speed * scale
        #             # arrows = self.pred_mesh.transform.apply_to(
        #             #     np.array([pn, end])
        #             # )
        #             self._arrow.set_data(pos=np.vstack([pn, end]), color=self.pred_mesh.color)
        #             self._arrow.visible = True
        #         else:
        #             self._arrow.visible = False

        if path is not None:
            centroid = np.mean(path, axis=0)
            self.stop_line.visible = y < 16
            # self.stop_sign.visible = y < 16
            # self.stop_sign_text.visible = y < 16
            self.caution_sign.visible = y < 16
            if y < 16:
                path = path[: int(len(path) * 2 / 3) - 1]
            self.path_line.set_data(pos=path)
            self.path_line.transform = transforms.MatrixTransform()
            self.path_line.transform.translate(-centroid)
            self.path_line.transform.rotate(180, (0, 0, 1))
            self.path_line.transform.translate(centroid)

        if gt_positions:
            pos3d = np.c_[np.array(gt_positions), np.full(len(gt_positions), self._GT_Z)]
            self.gt_dots.set_data(
                pos=pos3d, size=self.config.dot_size, face_color=(0, 1, 0, 1)
            )

        if histograms is not None:
            self.histogram_widget.update(histograms=histograms)

        if image is not None:
            self.camera_widget.update(image=image)

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


# -------------------- Script entry --------------------
if __name__ == "__main__":
    import time

    dash = CVPR25FullDashboard(
        CVPR25FullDashboardConfig(x_range=(0, 32), y_range=(0, 32))
    )
    dash.setup()

    path = np.array([[0, 0, 0], [10, 10, 0], [20, 5, 0], [30, 15, 0]])
    fake_hist = np.random.randint(0, 100, size=(END_BIN - START_BIN))
    start_time = time.time()

    try:
        while dash.is_okay:
            t = (time.time() - start_time) % 5
            y = 16 + 16 * np.sin(2 * np.pi * t / 5)
            position = (16.0, y)
            dash.update(frame=0, positions=[position], path=path, histograms=fake_hist)
    except KeyboardInterrupt:
        dash.close()
