from cc_hardware.utils import Component, config_wrapper
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import numpy as np
import matplotlib.cm as cm
import time

@config_wrapper
class ParticleFilterDashboardConfig:
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    zlim: tuple[float, float]
    xres: float
    yres: float
    zres: float
    num_x: int
    num_y: int
    num_z: int
    cam_z: float
    cmap: str = "hot"
    gamma: float = 1.0
    arc_span_deg: float = 30.0
    arc_pts: int = 60
    kf_q_pos: float = 2e-2
    kf_q_vel: float = 1e-4
    kf_r_pos: float = 0.5
    top_k_peaks: int = 20
    std_thresh: float = 0.3
    max_pos_var: float = 0.3
    overlay_w_frac: float = 0.25
    overlay_h_frac: float = 0.12
    overlay_margin_frac: float = 0.02

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
        if image.ndim == 2:
            norm = (image - image.min()) / (image.max() - image.min())
            image = (cm.hot(norm)[:, :, :3] * 255).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = np.interp(image, (image.min(), image.max()), (0, 255)).astype(np.uint8)
        h, w, _ = image.shape
        qimg = QtGui.QImage(image.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888)
        self.label.setPixmap(QtGui.QPixmap.fromImage(qimg))

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setBrush(QtGui.QColor(255, 255, 255, 100))
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.drawRoundedRect(self.rect(), 6, 6)

class TextWidget(QtWidgets.QWidget):
    def __init__(self, text: str, parent=None, font_size: int = 20):
        super().__init__(parent)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAutoFillBackground(False)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QtWidgets.QLabel(text)
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet(f"font-size: {font_size}px; font-weight: bold;")
        layout.addWidget(self.label)

class ParticleFilterDashboard(Component[ParticleFilterDashboardConfig]):
    def __init__(self, cfg: ParticleFilterDashboardConfig):
        super().__init__(cfg)

        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")
        pg.mkQApp()

        # === initialize the GUI window === #
        self.win = pg.GraphicsLayoutWidget(title="Volume Projection", show=True)

        # === Define top-down view === #
        wall_thickness = 0.1  # thinner
        self.top = self.win.addPlot()
        self.top.setAspectLocked()
        self.top.setLabels(bottom="X (m)", left="Z (m)")
        self.top.setXRange(cfg.xlim[0], cfg.xlim[1])
        self.top.setYRange(cfg.zlim[0]-wall_thickness, cfg.zlim[1])
        self.top.invertY(True)
        grey = pg.mkPen((200, 200, 200), width=1, style=QtCore.Qt.PenStyle.DashLine)
        self.top.addItem(pg.InfiniteLine(pos=0, angle=0, pen=grey))
        self.top.addItem(pg.InfiniteLine(pos=0, angle=90, pen=grey))

        # === Draw relay wall as black flat thin rectangle === #
        wall_mid = (cfg.zlim[0] + cfg.zlim[1]) / 2 # y value in plot
        wall_length = cfg.xlim[1] - cfg.xlim[0] # extent in x in plot
        

        # define rectangle as a QPainterPath
        rect = QtGui.QPainterPath() 
        rect.addRect(
            cfg.xlim[0], -wall_thickness, wall_length, wall_thickness
        )
        wall_item = QtWidgets.QGraphicsPathItem(rect)
        wall_item.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0)))
        wall_item.setPen(QtGui.QPen(QtCore.Qt.PenStyle.NoPen))
        self.top.addItem(wall_item)

        # === wall label bold, to right of wall === #
        wall_label = pg.TextItem("Relay Wall", anchor=(0.5, 0.5), color="k", angle=0)
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        wall_label.setFont(font)
        self.top.addItem(wall_label)
        wall_label.setPos(cfg.xlim[1] + 0.1, -wall_thickness / 2)

        # === plot particles as points === #
        self.particles = pg.ScatterPlotItem()
        self.top.addItem(self.particles)

        # === plot mean particle position as a point === #
        self.mean_particle = pg.ScatterPlotItem(size=30, pen=None, brush=pg.mkBrush(255, 0, 0, 255))
        self.top.addItem(self.mean_particle)

        # === plot past 10 mean particle positions as a line === #
        self.past_positions_x = []
        self.past_positions_z = []
        self.num_past_positions = 10
        # plot as line not as points
        
        self.past_mean_particles = pg.PlotCurveItem(pen=pg.mkPen(color=(255, 0, 0, 255), width=20))
        self.top.addItem(self.past_mean_particles)

        # === signal overlay QWidget === #
        self.sig_widget = CameraWidget(self.win)
        self.sig_label = TextWidget("Raw Signal", self.win, font_size=30)

        # === plot camera field of view === #
        self._box_sz = 0.1
        self.sensor_box = QtWidgets.QGraphicsRectItem(
            -self._box_sz * 0.2, -self._box_sz * 0.5, self._box_sz, self._box_sz
        )
        self.sensor_box.setBrush(QtGui.QBrush(QtGui.QColor(255, 0, 0)))
        self.sensor_box.setPen(QtGui.QPen(QtCore.Qt.PenStyle.NoPen))
        self.top.addItem(self.sensor_box)
        self.sensor_label = pg.TextItem("Sensor", anchor=(0.5, 0.0), color="r")
        lab_font = QtGui.QFont()
        lab_font.setPointSize(18); lab_font.setBold(True)
        self.sensor_label.setFont(lab_font)

        
        self.top.addItem(self.sensor_label)
        pen_fov = pg.mkPen((255, 0, 0), width=2,
                   style=QtCore.Qt.PenStyle.DashLine)

        self.fov_left  = QtWidgets.QGraphicsLineItem()
        self.fov_right = QtWidgets.QGraphicsLineItem()
        self.fov_left.setPen(pen_fov)
        self.fov_right.setPen(pen_fov)
        self.top.addItem(self.fov_left)
        self.top.addItem(self.fov_right)
        self.cam_z = cfg.cam_z

        # overlay geometry factors
        self.ov_wf = cfg.overlay_w_frac
        self.ov_hf = cfg.overlay_h_frac
        self.ov_mf = cfg.overlay_margin_frac

        # === show the window === #
        self.win.showFullScreen()
        self.sig_widget.show()
        self.sig_label.show()
        self._place_sig_widget()
        self.win.installEventFilter(self.win)

        # on window resize, reposition the camera widget
        orig_resize = self.win.resizeEvent
        def resizeEvent(ev):
            orig_resize(ev)
            self._place_sig_widget()
        self.win.resizeEvent = resizeEvent

    def _place_sig_widget(self):
        margin = 50
        h = self.win.size().height()
        w = self.win.size().width()
        self.sig_widget.move(
            w - self.sig_widget.width() - margin, h - self.sig_widget.height() - margin
        )
        self.sig_label.move(
            self.sig_widget.x(),
            self.sig_widget.y() - self.sig_label.height()
        )


    # ---- sensor pose from point-cloud ----
    def _update_sensor(self, pt_cloud: np.ndarray) -> None:
        pts = pt_cloud.reshape(-1, 3).astype(float)
        centroid = pts.mean(axis=0)
        pts -= centroid
        _, _, vh = np.linalg.svd(pts)
        normal = vh[2]

        z_axis = np.array([0.0, 0.0, 1.0])
        v = np.cross(normal, z_axis)
        c = float(np.dot(normal, z_axis))
        if np.linalg.norm(v) < 1e-6:
            R = np.eye(3)
        else:
            s = np.linalg.norm(v)
            vx = np.array(
                [[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]
            )
            R = np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)

        sensor_pos = -R @ centroid
        y, z = sensor_pos[1], abs(sensor_pos[2])

        cam_x = 0

        self.sensor_box.setPos(cam_x, self.cam_z)
        self.sensor_label.setPos(cam_x + self._box_sz, self.cam_z)

        ang_c = np.arctan2(y, z + 1e-9)          # central
        half   = np.deg2rad(45)                  # ±45° FOV

        for x_pos, item in ((np.min(pt_cloud[:, 0]), self.fov_left),
                        (np.max(pt_cloud[:, 0]), self.fov_right)):

            item.setVisible(True)
            item.setLine(x_pos, 0, 0, self.cam_z) # (a, b, c, d) line from (a, b) to (c, d)

        rot = -np.degrees(ang_c)
        self.sensor_box.setRotation(rot)
        # self.sensor_label.setAngle(-rot)

    # ---- main update ----
    def update(self, volume: np.ndarray, signal: np.ndarray, pt_cloud: np.ndarray) -> None:
        # === Update particle positions  === #
        self.particles.setData(volume[:, 0], volume[:, 2])

        # === Update mean particle position === #
        mean_particle = volume.mean(axis=0)
        self.mean_particle.setData([mean_particle[0]], [mean_particle[2]])

        # === Plot past 10 mean particle positions === #
        self.past_positions_x.append(mean_particle[0])
        self.past_positions_z.append(mean_particle[2])
        if len(self.past_positions_x) >= 10:
            self.past_positions_x.pop(0)
            self.past_positions_z.pop(0)
            
        self.past_mean_particles.setData(self.past_positions_x, self.past_positions_z)

        # === Update signal === #
        self.sig_widget.update(image=signal)

        # === Update sensor pose === #
        self._update_sensor(pt_cloud)

        pg.QtGui.QGuiApplication.processEvents()


    # ---- utility ----
    @property
    def is_okay(self):
        return self.win.isVisible()

    def close(self):
        self.win.close()
