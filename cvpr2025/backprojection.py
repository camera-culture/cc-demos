from copy import deepcopy
import time

import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch

from cc_hardware.drivers.spads import SPADDataType, SPADSensorConfig
from cc_hardware.utils import Component, config_wrapper
from cc_hardware.utils.constants import C

# ==========


def align_plane(pt_cloud: np.ndarray, dist_threshold=0.01) -> np.ndarray:
    pts = pt_cloud.reshape(-1, 3)
    centroid = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - centroid)
    normal = vh[2] / np.linalg.norm(vh[2])
    dists = np.dot(pts - centroid, normal)
    inliers = pts[np.abs(dists) < dist_threshold]
    centroid = inliers.mean(axis=0)
    _, _, vh = np.linalg.svd(inliers - centroid)
    normal = vh[2] / np.linalg.norm(vh[2])
    target = np.array([0, 0, 1.0])
    axis = np.cross(normal, target)
    if np.linalg.norm(axis) < 1e-6:
        R = np.eye(3)
    else:
        axis /= np.linalg.norm(axis)
        angle = np.arccos(np.dot(normal, target))
        K = np.array(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
        )
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    pts_trans = (pts - centroid) @ R.T
    mask = np.abs(pts_trans @ target) < dist_threshold
    plane_z = pts_trans[mask, 2].mean()
    pts_trans[:, 2] -= plane_z
    return pts_trans.reshape(pt_cloud.shape)


def extend_histograms(data, sensor_config, return_bins=False):
    assert SPADDataType.HISTOGRAM in data, "Histogram missing"
    histograms = data[SPADDataType.HISTOGRAM]
    start_bin = sensor_config.start_bin
    num_bins = sensor_config.num_bins
    subsample = sensor_config.subsample
    extended_bins = np.arange(0, start_bin + num_bins * subsample, subsample)
    extended_histograms = np.zeros(histograms.shape[:-1] + (len(extended_bins),))
    for i in range(histograms.shape[0]):
        for j in range(histograms.shape[1]):
            extended_histograms[i, j, -histograms.shape[2] :] = histograms[i, j]
    if return_bins:
        return extended_bins, extended_histograms
    return extended_histograms


def process_histograms(data, sensor_config):
    assert SPADDataType.HISTOGRAM in data, "Histogram missing"
    assert SPADDataType.DISTANCE in data, "Distance missing"
    histograms = data[SPADDataType.HISTOGRAM]
    distances = data[SPADDataType.DISTANCE]
    extended_bins, extended_histograms = extend_histograms(
        data, sensor_config, return_bins=True
    )
    processed_histograms = np.zeros_like(extended_histograms)
    for i in range(histograms.shape[0]):
        for j in range(histograms.shape[1]):
            distance = distances[i, j]
            distance_bin = distance / 1000 * 2 / C / sensor_config.timing_resolution
            closest_bin = np.argmin(np.abs(extended_bins - distance_bin))
            # subsample = sensor_config.subsample
            # adjusted_bins = (extended_bins - closest_bin * subsample)[closest_bin:]
            adjusted_hist = extended_histograms[i, j, closest_bin:]
            processed_histograms[i, j, : len(adjusted_hist)] = adjusted_hist
    return processed_histograms


@config_wrapper
class BackprojectionConfig:
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    z_range: tuple[float, float]
    num_x: int
    num_y: int
    num_z: int
    update_continuously: bool = False


class BackprojectionAlgorithm:
    def __init__(self, config: BackprojectionConfig, sensor_config: SPADSensorConfig):
        self.config = config
        self.sensor_config = sensor_config
        self.voxel_grid = self._create_voxel_grid()
        self.volume = np.zeros((self.voxel_grid.shape[0], 1))

    def _create_voxel_grid(self) -> np.ndarray:
        xr, yr, zr = self.config.x_range, self.config.y_range, self.config.z_range
        nx, ny, nz = self.config.num_x, self.config.num_y, self.config.num_z
        x = np.linspace(xr[0], xr[1], nx, endpoint=False)
        y = np.linspace(yr[0], yr[1], ny, endpoint=False)
        z = np.linspace(zr[0], zr[1], nz, endpoint=False)
        xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
        return np.stack([xv.ravel(), yv.ravel(), zv.ravel()], axis=-1)

    def update(self, data: dict[SPADDataType, np.ndarray]) -> np.ndarray:
        assert SPADDataType.POINT_CLOUD in data, "Point cloud missing"
        assert SPADDataType.HISTOGRAM in data, "Histogram missing"

        pt_cloud = data[SPADDataType.POINT_CLOUD].reshape(-1, 3)
        pt_cloud = align_plane(pt_cloud)
        # by default, histogram is 8x8xbins
        # need to duplicate each pixel so that it's (16x8)x(16x8)x(bins),
        # so 128x128xbins
        # data[SPADDataType.HISTOGRAM] = np.repeat(
        #     data[SPADDataType.HISTOGRAM], 16, axis=0
        # ).repeat(16, axis=1)
        # data[SPADDataType.DISTANCE] = np.repeat(
        #     data[SPADDataType.DISTANCE], 16, axis=0
        # ).repeat(16, axis=1)
        hists = process_histograms(data, self.sensor_config).reshape(
            pt_cloud.shape[0], -1
        )
        # argmax and set peaks to 1 and eveyrhing else to 0
        # peaks = np.argmax(hists, axis=1)
        # hists = np.zeros_like(hists)
        # hists[np.arange(hists.shape[0]), peaks] = 1

        bw = self.sensor_config.timing_resolution
        thresh = bw * C
        factor = bw * C / 2
        num_bins = hists.shape[1]
        cum_hists = np.cumsum(hists, axis=1)
        new_vol = np.zeros_like(self.volume)
        for i, pt in enumerate(pt_cloud):
            dists = np.linalg.norm(self.voxel_grid - pt, axis=1)
            cum = cum_hists[i]
            lower = np.clip(
                np.ceil((dists - thresh) / factor).astype(int), 0, num_bins - 1
            )
            upper = np.clip(
                np.floor((dists + thresh) / factor).astype(int), 0, num_bins - 1
            )
            sums = cum[upper] - np.where(lower > 0, cum[lower - 1], 0)
            new_vol += sums.reshape(-1, 1)
        if self.config.update_continuously:
            self.volume += new_vol
            out = self.volume
        else:
            out = new_vol
        return out.reshape(self.config.num_x, self.config.num_y, self.config.num_z)

    @property
    def resolution(self) -> tuple[float, float, float]:
        """Returns the resolution of the voxel grid."""
        x_res = (self.config.x_range[1] - self.config.x_range[0]) / self.config.num_x
        y_res = (self.config.y_range[1] - self.config.y_range[0]) / self.config.num_y
        z_res = (self.config.z_range[1] - self.config.z_range[0]) / self.config.num_z
        return x_res, y_res, z_res

    @property
    def xres(self) -> float:
        """Returns the x resolution of the voxel grid."""
        return self.resolution[0]

    @property
    def yres(self) -> float:
        """Returns the y resolution of the voxel grid."""
        return self.resolution[1]

    @property
    def zres(self) -> float:
        """Returns the z resolution of the voxel grid."""
        return self.resolution[2]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def align_plane(pt_cloud: torch.Tensor, dist_threshold: float = 0.01) -> torch.Tensor:
    pts = pt_cloud.view(-1, 3).to(device)
    centroid = pts.mean(0, keepdim=True)
    _, _, vh = torch.linalg.svd(pts - centroid, full_matrices=False)
    normal = vh[-1] / torch.linalg.norm(vh[-1])
    dists = (pts - centroid) @ normal
    inliers = pts[torch.abs(dists) < dist_threshold]
    centroid = inliers.mean(0, keepdim=True)
    _, _, vh = torch.linalg.svd(inliers - centroid, full_matrices=False)
    normal = vh[-1] / torch.linalg.norm(vh[-1])
    target = torch.tensor([0.0, 0.0, 1.0], device=device)
    axis = torch.linalg.cross(normal, target)
    if torch.linalg.norm(axis) < 1e-6:
        R = torch.eye(3, device=device)
    else:
        axis = axis / torch.linalg.norm(axis)
        angle = torch.arccos(torch.clamp(torch.dot(normal, target), -1.0, 1.0))
        K = torch.tensor(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]],
            device=device,
        )
        R = (
            torch.eye(3, device=device)
            + torch.sin(angle) * K
            + (1 - torch.cos(angle)) * (K @ K)
        )
    pts_trans = (pts - centroid) @ R.T
    mask = torch.abs(pts_trans @ target) < dist_threshold
    pts_trans[:, 2] -= pts_trans[mask, 2].mean()
    return pts_trans.view_as(pt_cloud)


def extend_histograms(data, cfg: SPADSensorConfig, return_bins=False):
    h = torch.as_tensor(data[SPADDataType.HISTOGRAM], device=device)
    bins = torch.arange(
        0, cfg.start_bin + cfg.num_bins * cfg.subsample, cfg.subsample, device=device
    )
    ext = torch.zeros(*h.shape[:-1], len(bins), device=device)
    ext[..., -h.shape[-1] :] = h
    return (bins, ext) if return_bins else ext


def process_histograms(data, cfg: SPADSensorConfig):
    h = torch.as_tensor(data[SPADDataType.HISTOGRAM], device=device)
    d = torch.as_tensor(data[SPADDataType.DISTANCE], device=device)
    bins, ext = extend_histograms(data, cfg, True)
    proc = torch.zeros_like(ext)
    dist_bins = d / 1000 * 2 / C / cfg.timing_resolution
    idx = (bins.unsqueeze(0).unsqueeze(0) - dist_bins.unsqueeze(-1)).abs().argmin(-1)
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            k = idx[i, j]
            adj = ext[i, j, k:]
            proc[i, j, : adj.shape[0]] = adj
    return proc


# ---------- algorithm ---------- #


class BackprojectionAlgorithm:
    def __init__(self, cfg: BackprojectionConfig, sensor_config: SPADSensorConfig):
        self.cfg = cfg
        self.sensor_cfg = sensor_config
        self.device = device
        self.voxel_grid = self._create_voxel_grid().to(device)
        self.volume = torch.zeros(
            (self.voxel_grid.shape[0], 1), device=device, dtype=torch.float32
        )

    def _create_voxel_grid(self):
        xr, yr, zr = self.cfg.x_range, self.cfg.y_range, self.cfg.z_range
        nx, ny, nz = self.cfg.num_x, self.cfg.num_y, self.cfg.num_z
        x = torch.linspace(
            xr[0], xr[1], nx, device=device, dtype=torch.float32, requires_grad=False
        )
        y = torch.linspace(
            yr[0], yr[1], ny, device=device, dtype=torch.float32, requires_grad=False
        )
        z = torch.linspace(
            zr[0], zr[1], nz, device=device, dtype=torch.float32, requires_grad=False
        )
        xv, yv, zv = torch.meshgrid(x, y, z, indexing="ij")
        return torch.stack((xv, yv, zv), -1).view(-1, 3)

    def update(self, data):
        pc = torch.as_tensor(
            data[SPADDataType.POINT_CLOUD], device=device, dtype=torch.float32
        ).view(-1, 3)
        pc = align_plane(pc)
        h = process_histograms(data, self.sensor_cfg).view(pc.shape[0], -1)
        bw = self.sensor_cfg.timing_resolution
        thresh, factor = bw * C, bw * C / 2
        cum = h.cumsum(-1)
        dists = torch.cdist(pc, self.voxel_grid)  # (P, V)
        lower = ((dists - thresh) / factor).ceil().clamp(0, h.shape[1] - 1).long()
        upper = ((dists + thresh) / factor).floor().clamp(0, h.shape[1] - 1).long()
        sums_u = cum.gather(1, upper)
        sums_l = cum.gather(1, (lower - 1).clamp(0))  # safe for lower==0
        sums = torch.where(lower > 0, sums_u - sums_l, sums_u)  # (P, V)
        new_vol = sums.sum(0, keepdim=True).t()  # (V, 1)
        if self.cfg.update_continuously:
            self.volume += new_vol
            out = self.volume
        else:
            out = new_vol
        return out.view(self.cfg.num_x, self.cfg.num_y, self.cfg.num_z).cpu().numpy()

    @property
    def resolution(self) -> tuple[float, float, float]:
        """Returns the resolution of the voxel grid."""
        x_res = (self.cfg.x_range[1] - self.cfg.x_range[0]) / self.cfg.num_x
        y_res = (self.cfg.y_range[1] - self.cfg.y_range[0]) / self.cfg.num_y
        z_res = (self.cfg.z_range[1] - self.cfg.z_range[0]) / self.cfg.num_z
        return x_res, y_res, z_res

    @property
    def xres(self) -> float:
        """Returns the x resolution of the voxel grid."""
        return self.resolution[0]

    @property
    def yres(self) -> float:
        """Returns the y resolution of the voxel grid."""
        return self.resolution[1]

    @property
    def zres(self) -> float:
        """Returns the z resolution of the voxel grid."""
        return self.resolution[2]

    @property
    def config(self) -> BackprojectionConfig:
        """Returns the configuration of the algorithm."""
        return self.cfg


# ==========

# @config_wrapper
# class BackprojectionDashboardConfig:
#     xlim: tuple[float, float]
#     ylim: tuple[float, float]
#     zlim: tuple[float, float]
#     xres: float
#     yres: float
#     zres: float
#     num_x: int
#     num_y: int
#     num_z: int
#     num_cols: int = 5
#     cmap: str = "hot"
#     gamma: float = 2.0
#     show_projection: bool = True


# class BackprojectionDashboard(Component[BackprojectionDashboardConfig]):
#     def __init__(self, config: BackprojectionDashboardConfig):
#         super().__init__(config)
#         plt.ion()
#         norm = mcolors.PowerNorm(gamma=config.gamma)
#         self.kwargs = {"cmap": config.cmap, "norm": norm}

#         def _format_ax(ax, axis, shape):
#             xnum, ynum = shape
#             if axis == "x":
#                 xlim, ylim = config.zlim, config.ylim
#                 xlabel, ylabel = "Z (m)", "Y (m)"
#             elif axis == "y":
#                 xlim, ylim = config.zlim, config.xlim
#                 xlabel, ylabel = "Z (m)", "X (m)"
#             else:
#                 xlim, ylim = config.xlim, config.ylim
#                 xlabel, ylabel = "X (m)", "Y (m)"
#             xticks = np.linspace(0, xnum - 1, 5)
#             xlabels = np.linspace(xlim[0], xlim[1], 5)
#             yticks = np.linspace(0, ynum - 1, 5)
#             ylabels = np.linspace(ylim[0], ylim[1], 5)
#             ax.set_xticks(xticks)
#             ax.set_xticklabels([f"{v:.2f}" for v in xlabels])
#             ax.set_yticks(yticks)
#             ax.set_yticklabels([f"{v:.2f}" for v in ylabels])
#             ax.set_xlabel(xlabel)
#             ax.set_ylabel(ylabel)

#         # Projection
#         if config.show_projection:
#             n_proj = 4
#             self.fig_proj = plt.figure(figsize=(4 * n_proj, 4))
#             self.proj_axes = []

#             # Y-Z
#             ax1 = self.fig_proj.add_subplot(1, n_proj, 1)
#             im1 = ax1.imshow(
#                 np.zeros((config.num_y, config.num_z)).T, **deepcopy(self.kwargs)
#             )
#             ax1.set_title("Y-Z")
#             _format_ax(ax1, "x", (config.num_z, config.num_y))
#             self.proj_axes.append((ax1, im1))

#             # X-Z
#             ax2 = self.fig_proj.add_subplot(1, n_proj, 2)
#             im2 = ax2.imshow(
#                 np.zeros((config.num_x, config.num_z)).T, **deepcopy(self.kwargs)
#             )
#             ax2.set_title("X-Z")
#             _format_ax(ax2, "y", (config.num_z, config.num_x))
#             self.proj_axes.append((ax2, im2))

#             # X-Y
#             ax3 = self.fig_proj.add_subplot(1, n_proj, 3)
#             im3 = ax3.imshow(
#                 np.zeros((config.num_x, config.num_y)).T, **deepcopy(self.kwargs)
#             )
#             ax3.set_title("X-Y")
#             _format_ax(ax3, "z", (config.num_y, config.num_x))
#             self.proj_axes.append((ax3, im3))

#             # Signal
#             ax4 = self.fig_proj.add_subplot(1, n_proj, 4)
#             im4 = ax4.imshow(np.zeros((1, 1)), **deepcopy(self.kwargs))
#             ax4.set_title("Signal")
#             self.proj_axes.append((ax4, im4))

#             self.fig_proj.suptitle("Volume Projection")
#             self.fig_proj.canvas.manager.full_screen_toggle()

#     def update(self, volume: np.ndarray, signal: np.ndarray) -> None:
#         xs = np.max(volume, axis=0)
#         ys = np.max(volume, axis=1)
#         zs = np.max(volume, axis=2)
#         min_val = min(xs.min(), ys.min(), zs.min())
#         max_val = max(xs.max(), ys.max(), zs.max())

#         for (ax, im), arr in zip(self.proj_axes[:3], [xs, ys, zs]):
#             im.set_data(arr)
#             im.set_clim([min_val, max_val])

#         ax, im = self.proj_axes[-1]
#         im.set_data(signal)
#         im.set_clim([signal.min(), signal.max()])

#         self.fig_proj.canvas.draw_idle()
#         plt.pause(0.001)

#     @property
#     def is_okay(self) -> bool:
#         return plt.fignum_exists(self.fig_proj.number)

#     def close(self) -> None:
#         plt.close(self.fig_proj)

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


@config_wrapper
class BackprojectionDashboardConfig:
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    zlim: tuple[float, float]
    xres: float
    yres: float
    zres: float
    num_x: int
    num_y: int
    num_z: int
    num_cols: int = 5
    cmap: str = "hot"
    gamma: float = 2.0
    show_projection: bool = True


class BackprojectionDashboard(Component[BackprojectionDashboardConfig]):
    def __init__(self, config: BackprojectionDashboardConfig):
        super().__init__(config)

        pg.mkQApp()
        self.win = pg.GraphicsLayoutWidget(title="Volume Projection", show=True)
        self.win.showMaximized()

        lut = pg.colormap.get(config.cmap, source="matplotlib").getLookupTable()
        self.proj_items = []

        def _add_img(row: int, col: int, title: str, shape: tuple[int, int]):
            plt_item = self.win.addPlot(row=row, col=col)
            plt_item.setTitle(title)
            plt_item.hideAxis("right")
            plt_item.hideAxis("top")
            plt_item.setAspectLocked()

            img = pg.ImageItem(np.zeros(shape, dtype=np.float32))
            img.setLookupTable(lut)
            plt_item.addItem(img)
            return img

        if config.show_projection:
            imgs = [
                _add_img(0, 0, "Y-Z", (config.num_z, config.num_y)),
                _add_img(0, 1, "X-Z", (config.num_z, config.num_x)),
                _add_img(0, 2, "X-Y", (config.num_y, config.num_x)),
                _add_img(0, 3, "Signal", (1, 1)),
            ]
            self.proj_items = imgs

    def update(self, volume: np.ndarray, signal: np.ndarray) -> None:
        xs = np.max(volume, axis=0)
        xs[self.config.num_y // 2 :, :] = 0.0  # zero out lower half
        ys = np.max(volume, axis=1)
        zs = np.max(volume, axis=2)
        vmin = min(xs.min(), ys.min(), zs.min())
        vmax = max(xs.max(), ys.max(), zs.max())

        imgs = [xs.T, ys.T, zs.T]
        for img_item, arr in zip(self.proj_items, imgs[:3]):
            img_item.setImage(arr.astype(np.float32), levels=(vmin, vmax))

        self.proj_items[-1].setImage(
            signal.astype(np.float32).T, levels=(signal.min(), signal.max())
        )

        pg.QtGui.QGuiApplication.processEvents()

    @property
    def is_okay(self) -> bool:
        return self.win.isVisible()

    def close(self) -> None:
        self.win.close()


# ============================================================
# CONFIG
# ============================================================
@config_wrapper
class FinalBackprojectionDashboardConfig:
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    zlim: tuple[float, float]
    xres: float
    yres: float
    zres: float
    num_x: int
    num_y: int
    num_z: int
    cmap: str = "hot"
    gamma: float = 2.0
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


# ============================================================
# DASHBOARD
# ============================================================
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

class FinalBackprojectionDashboard(Component[FinalBackprojectionDashboardConfig]):
    def __init__(self, cfg: FinalBackprojectionDashboardConfig):
        super().__init__(cfg)

        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")
        pg.mkQApp()

        # ---- window ----
        self.win = pg.GraphicsLayoutWidget(title="Volume Projection", show=True)

        # ---- top-down ----
        self.top = self.win.addPlot()
        self.top.setAspectLocked()
        self.top.setLabels(bottom="Z (m)", left="Y (m)")
        self.top.setXRange(cfg.zlim[0], cfg.zlim[1])
        self.top.setYRange(cfg.ylim[0], cfg.ylim[1])
        self.top.invertY(True)
        grey = pg.mkPen((200, 200, 200), width=1, style=QtCore.Qt.PenStyle.DashLine)
        self.top.addItem(pg.InfiniteLine(pos=0, angle=0, pen=grey))
        self.top.addItem(pg.InfiniteLine(pos=0, angle=90, pen=grey))

        # ---- wall (black flat thin rectangle) ----
        y_mid = (cfg.ylim[0] + cfg.ylim[1]) * 0.5
        wall_half = 0.5
        wall_thickness = 0.1  # thinner

        rect = QtGui.QPainterPath()
        rect.addRect(
            -wall_thickness * 0.5, y_mid - wall_half, wall_thickness, wall_half * 2
        )
        wall_item = QtWidgets.QGraphicsPathItem(rect)
        wall_item.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0)))
        wall_item.setPen(QtGui.QPen(QtCore.Qt.PenStyle.NoPen))
        self.top.addItem(wall_item)

        # ---- wall label rotated, bold, to left of wall ----
        wall_label = pg.TextItem("Relay Wall", anchor=(0.5, 0.5), color="k", angle=90)
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        wall_label.setFont(font)
        self.top.addItem(wall_label)
        wall_label.setPos(-wall_thickness, y_mid)

        # ---- arc ----
        self.arc = pg.ScatterPlotItem()
        self.top.addItem(self.arc)

        # ---- signal overlay QWidget ----
        self.sig_widget = CameraWidget(self.win)
        self.sig_label = TextWidget("Raw Signal", self.win, font_size=30)

        # pt cloud stuff
        self._box_sz = 0.1
        self.sensor_box = QtWidgets.QGraphicsRectItem(
            -self._box_sz * 0.5, -self._box_sz * 0.5, self._box_sz, self._box_sz
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

        # overlay geometry factors
        self.ov_wf = cfg.overlay_w_frac
        self.ov_hf = cfg.overlay_h_frac
        self.ov_mf = cfg.overlay_margin_frac

        # ---- constants & state ----
        self.arc_half = np.deg2rad(cfg.arc_span_deg) * 0.5
        self.y_span = cfg.ylim[1] - cfg.ylim[0]
        self.z_span = cfg.zlim[1] - cfg.zlim[0]
        self.kf_state = None
        self.kf_cov = None
        self.t_last = None
        self.certs = [0] * 20

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

    # ---- helpers ----
    def eventFilter(self, obj, ev):
        if ev.type() == QtCore.QEvent.Type.Resize and obj is self.win:
            self._place_sig_widget()
        return super().eventFilter(obj, ev)

    def _arc_pts(self, r, th):
        n = self.config.arc_pts
        ang = np.linspace(th - self.arc_half, th + self.arc_half, n)
        xs, ys = r * np.cos(ang), r * np.sin(ang)
        mid = n // 2
        sizes = np.hstack(
            (np.linspace(8, 48, mid), np.linspace(48, 8, n - mid))
        ).astype(int)
        alpha = np.hstack(
            (np.linspace(160, 255, mid), np.linspace(255, 160, n - mid))
        ).astype(int)
        return xs, ys, sizes.tolist(), alpha.tolist()

    def _kalman(self, y_m, z_m):
        c = self.config
        t = time.time()
        if self.kf_state is None:
            self.kf_state = np.array([y_m, z_m, 0.0, 0.0])
            self.kf_cov = np.eye(4) * 1e-2
            self.t_last = t
            return self.kf_state[:2]
        dt = max(t - self.t_last, 1e-3)
        A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        Q = np.diag([c.kf_q_pos, c.kf_q_pos, c.kf_q_vel, c.kf_q_vel])
        self.kf_state = A @ self.kf_state
        self.kf_cov = A @ self.kf_cov @ A.T + Q
        z = np.array([y_m, z_m])
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        R = np.eye(2) * c.kf_r_pos
        S = H @ self.kf_cov @ H.T + R
        K = self.kf_cov @ H.T @ np.linalg.inv(S)
        self.kf_state += K @ (z - H @ self.kf_state)
        self.kf_cov = (np.eye(4) - K @ H) @ self.kf_cov
        self.t_last = t
        return self.kf_state[:2]

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

        self.sensor_box.setPos(z, y)
        self.sensor_label.setPos(z, y + self._box_sz)

        ang_c = np.arctan2(y, z + 1e-9)          # central
        half   = np.deg2rad(45)                  # ±45° FOV

        for ang, item in ((ang_c - half, self.fov_left),
                        (ang_c + half, self.fov_right)):
            c = np.cos(ang)
            if abs(c) < 1e-3:                     # avoid div 0
                item.setVisible(False)
                continue
            t = -z / c                            # hit z=0 wall
            item.setVisible(True)
            item.setLine(z, y, 0.0, y + t*np.sin(ang))

        rot = -np.degrees(ang_c)
        self.sensor_box.setRotation(rot)
        self.sensor_label.setAngle(-rot)

    # ---- main update ----
    def update(self, volume: np.ndarray, signal: np.ndarray, pt_cloud: np.ndarray) -> None:
        cfg = self.config
        yz = np.max(volume, axis=0)

        flat = yz.ravel()
        k = min(cfg.top_k_peaks, flat.size)
        idx = np.argpartition(flat, -k)[-k:]
        vals = flat[idx].astype(float)
        wts = vals / vals.sum()

        y_idx, z_idx = np.unravel_index(idx, yz.shape)
        y_coords = cfg.ylim[0] + (y_idx + 0.5) * self.y_span / cfg.num_y
        z_coords = cfg.zlim[0] + (z_idx + 0.5) * self.z_span / cfg.num_z
        y_m, z_m = float(np.dot(wts, y_coords)), float(np.dot(wts, z_coords))

        radii = np.hypot(z_coords, y_coords)
        sigma_r = float(np.sqrt(np.dot(wts, (radii - np.dot(wts, radii)) ** 2)))

        y_f, z_f = self._kalman(y_m, z_m)
        pos_var = float(self.kf_cov[0, 0] + self.kf_cov[1, 1])

        cert = max(0, 1 - sigma_r / cfg.std_thresh) * max(
            0, 1 - pos_var / cfg.max_pos_var
        )

        self.certs.append(cert)
        if len(self.certs) > 20:
            self.certs.pop(0)
        cert = np.mean(self.certs)

        if cert < 0.1:
            self.arc.setData([], [])
        else:
            r_f = np.hypot(z_f, y_f)
            th_f = np.arctan2(y_f, z_f)
            xs, ys, sz, base_a = self._arc_pts(r_f, th_f)
            alphas = (cert * np.array(base_a)).clip(1, 255).astype(int)
            brushes = [pg.mkBrush(0, 128, 255, a) for a in alphas]
            self.arc.setData(xs, ys, size=sz, brush=brushes, pen=None)

        self.sig_widget.update(image=signal)
        self._update_sensor(pt_cloud)

        pg.QtGui.QGuiApplication.processEvents()

    # ---- utility ----
    @property
    def is_okay(self):
        return self.win.isVisible()

    def close(self):
        self.win.close()
