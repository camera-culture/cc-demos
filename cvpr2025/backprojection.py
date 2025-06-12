from copy import deepcopy

import matplotlib.colors as mcolors
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
            [[0, -axis[2], axis[1]],
             [axis[2], 0, -axis[0]],
             [-axis[1], axis[0], 0]], device=device
        )
        R = torch.eye(3, device=device) + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
    pts_trans = (pts - centroid) @ R.T
    mask = torch.abs(pts_trans @ target) < dist_threshold
    pts_trans[:, 2] -= pts_trans[mask, 2].mean()
    return pts_trans.view_as(pt_cloud)


def extend_histograms(data, cfg: SPADSensorConfig, return_bins=False):
    h = torch.as_tensor(data[SPADDataType.HISTOGRAM], device=device)
    bins = torch.arange(0, cfg.start_bin + cfg.num_bins * cfg.subsample, cfg.subsample,
                        device=device)
    ext = torch.zeros(*h.shape[:-1], len(bins), device=device)
    ext[..., -h.shape[-1]:] = h
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
            proc[i, j, :adj.shape[0]] = adj
    return proc


# ---------- algorithm ---------- #

class BackprojectionAlgorithm:
    def __init__(self, cfg: BackprojectionConfig, sensor_config: SPADSensorConfig):
        self.cfg = cfg
        self.sensor_cfg = sensor_config
        self.device = device
        self.voxel_grid = self._create_voxel_grid().to(device)
        self.volume = torch.zeros((self.voxel_grid.shape[0], 1), device=device, dtype=torch.float32)

    def _create_voxel_grid(self):
        xr, yr, zr = self.cfg.x_range, self.cfg.y_range, self.cfg.z_range
        nx, ny, nz = self.cfg.num_x, self.cfg.num_y, self.cfg.num_z
        x = torch.linspace(xr[0], xr[1], nx, device=device, dtype=torch.float32, requires_grad=False)
        y = torch.linspace(yr[0], yr[1], ny, device=device, dtype=torch.float32, requires_grad=False)
        z = torch.linspace(zr[0], zr[1], nz, device=device, dtype=torch.float32, requires_grad=False)
        xv, yv, zv = torch.meshgrid(x, y, z, indexing="ij")
        return torch.stack((xv, yv, zv), -1).view(-1, 3)

    def update(self, data):
        pc = torch.as_tensor(data[SPADDataType.POINT_CLOUD], device=device, dtype=torch.float32).view(-1, 3)
        pc = align_plane(pc)
        h = process_histograms(data, self.sensor_cfg).view(pc.shape[0], -1)
        bw = self.sensor_cfg.timing_resolution
        thresh, factor = bw * C, bw * C / 2
        cum = h.cumsum(-1)
        dists = torch.cdist(pc, self.voxel_grid)               # (P, V)
        lower = ((dists - thresh) / factor).ceil().clamp(0, h.shape[1] - 1).long()
        upper = ((dists + thresh) / factor).floor().clamp(0, h.shape[1] - 1).long()
        sums_u = cum.gather(1, upper)
        sums_l = cum.gather(1, (lower - 1).clamp(0))           # safe for lower==0
        sums = torch.where(lower > 0, sums_u - sums_l, sums_u) # (P, V)
        new_vol = sums.sum(0, keepdim=True).t()                # (V, 1)
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
    show_slices: bool = False
    show_projection: bool = True
    show_images: bool = False


class BackprojectionDashboard(Component[BackprojectionDashboardConfig]):
    def __init__(self, config: BackprojectionDashboardConfig):
        super().__init__(config)
        plt.ion()
        norm = mcolors.PowerNorm(gamma=config.gamma)
        self.kwargs = {"cmap": config.cmap, "norm": norm}

        def _format_ax(ax, axis, shape):
            xnum, ynum = shape
            if axis == "x":
                xlim, ylim = config.zlim, config.ylim
                xlabel, ylabel = "Z (m)", "Y (m)"
            elif axis == "y":
                xlim, ylim = config.zlim, config.xlim
                xlabel, ylabel = "Z (m)", "X (m)"
            else:
                xlim, ylim = config.xlim, config.ylim
                xlabel, ylabel = "X (m)", "Y (m)"
            xticks = np.linspace(0, xnum - 1, 5)
            xlabels = np.linspace(xlim[0], xlim[1], 5)
            yticks = np.linspace(0, ynum - 1, 5)
            ylabels = np.linspace(ylim[0], ylim[1], 5)
            ax.set_xticks(xticks)
            ax.set_xticklabels([f"{v:.2f}" for v in xlabels])
            ax.set_yticks(yticks)
            ax.set_yticklabels([f"{v:.2f}" for v in ylabels])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        # Slices
        if config.show_slices:
            self.fig_slices = plt.figure(
                figsize=(
                    3 * config.num_cols,
                    3 * int(np.ceil(config.num_x / config.num_cols)),
                )
            )
            self.slice_imgs = {axis: [] for axis in ["x", "y", "z"]}
            for axis in ["x", "y", "z"]:
                n = getattr(config, f"num_{axis}")
                rows = int(np.ceil(n / config.num_cols))
                for i in range(n):
                    ax = self.fig_slices.add_subplot(rows, config.num_cols, i + 1)
                    shape = (
                        (config.num_y, config.num_z)
                        if axis == "x"
                        else (
                            (config.num_x, config.num_z)
                            if axis == "y"
                            else (config.num_x, config.num_y)
                        )
                    )
                    im = ax.imshow(np.zeros(shape).T, **deepcopy(self.kwargs))
                    # ax.set_title(
                    #     f"{axis} = {config.__dict__[f'{axis}lim'][0] + i * config.__dict__[f'{axis}res']:.2f}"
                    # )
                    _format_ax(ax, axis, shape)
                    self.slice_imgs[axis].append((ax, im))
            self.fig_slices.suptitle("Volume Slices")
            self.fig_slices.show()

        if config.show_images:
            self.fig_images = plt.figure(figsize=(16, 16))
            for i in range(4):
                for j in range(4):
                    ax = self.fig_images.add_subplot(4, 4, i * 4 + j + 1)
                    ax.imshow(
                        np.zeros((config.num_y, config.num_x)).T,
                        **deepcopy(self.kwargs),
                    )
                    ax.set_title(f"Image {i * 4 + j + 1}")
                    # _format_ax(ax, "z", (config.num_y, config.num_x))

        # Projection
        if config.show_projection:
            n_proj = 4
            self.fig_proj = plt.figure(figsize=(4 * n_proj, 4))
            self.proj_axes = []

            # Y-Z
            ax1 = self.fig_proj.add_subplot(1, n_proj, 1)
            shape1 = (config.num_z, config.num_y)
            im1 = ax1.imshow(
                np.zeros((config.num_y, config.num_z)).T, **deepcopy(self.kwargs)
            )
            ax1.set_title("Y-Z")
            _format_ax(ax1, "x", shape1)
            self.proj_axes.append((ax1, im1))

            # X-Z
            ax2 = self.fig_proj.add_subplot(1, n_proj, 2)
            shape2 = (config.num_z, config.num_x)
            im2 = ax2.imshow(
                np.zeros((config.num_x, config.num_z)).T, **deepcopy(self.kwargs)
            )
            ax2.set_title("X-Z")
            _format_ax(ax2, "y", shape2)
            self.proj_axes.append((ax2, im2))

            # X-Y
            ax3 = self.fig_proj.add_subplot(1, n_proj, 3)
            shape3 = (config.num_y, config.num_x)
            im3 = ax3.imshow(
                np.zeros((config.num_x, config.num_y)).T, **deepcopy(self.kwargs)
            )
            ax3.set_title("X-Y")
            _format_ax(ax3, "z", shape3)
            self.proj_axes.append((ax3, im3))

            # Signal
            ax4 = self.fig_proj.add_subplot(1, n_proj, 4)
            im4 = ax4.imshow(np.zeros((1, 1)), **deepcopy(self.kwargs))
            ax4.set_title("Signal")
            self.proj_axes.append((ax4, im4))

            self.fig_proj.suptitle("Volume Projection")
            # show fullscreen
            self.fig_proj.canvas.manager.full_screen_toggle()

    def update(self, volume: np.ndarray, signal: np.ndarray) -> None:
        cfg = self.config
        if cfg.show_slices:
            for axis in ["x", "y", "z"]:
                for i, (ax, im) in enumerate(self.slice_imgs[axis]):
                    sl = (
                        volume[i].T
                        if axis == "x"
                        else volume[:, i, :].T
                        if axis == "y"
                        else volume[:, :, i].T
                    )
                    im.set_data(sl)
                    im.set_clim([sl.min(), sl.max()])
            self.fig_slices.canvas.draw_idle()

        if cfg.show_projection:
            xs = np.max(volume, axis=0)
            ys = np.max(volume, axis=1)
            zs = np.max(volume, axis=2)
            min_val = min(xs.min(), ys.min(), zs.min())
            max_val = max(xs.max(), ys.max(), zs.max())
            for (ax, im), arr in zip(self.proj_axes[:3], [xs, ys, zs]):
                im.set_data(arr)
                im.set_clim([min_val, max_val])
            ax, im = self.proj_axes[-1]
            im.set_data(signal)
            im.set_clim([signal.min(), signal.max()])
            self.fig_proj.canvas.draw_idle()

        if cfg.show_images:
            hist = signal[:, :16]
            hist = hist.reshape(64, 4, 4)
            for i in range(4):
                for j in range(4):
                    ax = self.fig_images.axes[i * 4 + j]
                    im = ax.images[0]
                    im.set_data(hist[:, i, j].reshape(8, 8))
                    im.set_clim([hist.min(), hist.max()])
            self.fig_images.canvas.draw_idle()
        plt.pause(0.001)

    @property
    def is_okay(self) -> bool:
        """Returns True if the dashboard is okay, i.e., not closed."""
        okay = True
        if self.config.show_slices:
            okay &= plt.fignum_exists(self.fig_slices.number)
        if self.config.show_projection:
            okay &= plt.fignum_exists(self.fig_proj.number)
        return okay

    def close(self) -> None:
        """Closes the dashboard."""
        if self.config.show_slices:
            plt.close(self.fig_slices)
        if self.config.show_projection:
            plt.close(self.fig_proj)
