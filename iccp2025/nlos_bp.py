import time
from datetime import datetime
from functools import partial
from pathlib import Path
import numpy as np
import tqdm
import matplotlib.pyplot as plt

from process import fit_plane_to_pt_clouds, remove_1b
from visualization import (
    ParticleFilterDashboardConfig,
    ParticleFilterDashboard,
)

from reconstruction import (
    ParticleFilterAlgorithm, 
    ParticleFilterConfig,
    CanonConfig,
)

from cc_hardware.drivers.spads import SPADDataType, SPADSensor, SPADSensorConfig
from cc_hardware.drivers.spads.spad_wrappers import SPADMovingAverageWrapperConfig

from cc_hardware.drivers.spads.vl53l8ch import RangingMode, VL53L8CHConfig4x4, VL53L8CHConfig8x8
from cc_hardware.tools.dashboard.spad_dashboard import (
    DummySPADDashboardConfig,
    SPADDashboard,
    SPADDashboardConfig,
)
from cc_hardware.tools.dashboard.spad_dashboard.pyqtgraph import (
    PyQtGraphDashboardConfig,
)
from cc_hardware.utils import Manager, get_logger, register_cli, run_cli
from cc_hardware.utils.file_handlers import PklHandler

# ==================== SET PARAMETERS HERE ==================== #
debug = True
RECORD = True
RECAPTURE_BACKGROUND = True

WRAPPED_SENSOR_CONFIG = VL53L8CHConfig4x4.create(
    num_bins=48,
    subsample=1,
    start_bin=30, # 16
    ranging_mode=RangingMode.CONTINUOUS,
    ranging_frequency_hz=30,
    data_type=SPADDataType.HISTOGRAM | SPADDataType.POINT_CLOUD | SPADDataType.DISTANCE,
)

WRAPPED_SENSOR = SPADMovingAverageWrapperConfig.create(
    wrapped=WRAPPED_SENSOR_CONFIG,
    window_size=1,
)

# ================= DO NOT EDIT BELOW THIS LINE ================= #

NOW = datetime.now()

if debug: 
    LOGDIR: Path = Path("logs") / "debug"
else: 
    LOGDIR: Path = Path("logs") / NOW.strftime("%Y-%m-%d") / NOW.strftime("%H-%M-%S")
OUTPUT_PKL: Path = LOGDIR / "data.pkl"


# === Set up sensor and dashboard === #
SENSOR = WRAPPED_SENSOR
DASHBOARD = PyQtGraphDashboardConfig.create(fullscreen=True)
DASHBOARD = DummySPADDashboardConfig.create()

if RECORD: 
    LOGDIR.mkdir(exist_ok=True, parents=True)

global START_FRAME 
START_FRAME = WRAPPED_SENSOR_CONFIG.start_bin

# === Load point cloud from calibration file === #
pt_cloud_file = Path("logs") / "pt_cloud" / "calibration.npz"
global PT_CLOUD_GLOBAL, CAM_Z, BIN_0
PT_CLOUD_GLOBAL = np.load(pt_cloud_file)['pt_cloud']
CAM_Z = np.load(pt_cloud_file)['cam_z']
BIN_0 = np.load(pt_cloud_file)['bin_0']

# === Load background if available === #
global BACKGROUND, HIST_MASK
background_file = Path("logs") / "background" / "calibration.npz"
if not RECAPTURE_BACKGROUND:
    BACKGROUND = np.load(background_file)['background']
    HIST_MASK = np.load(background_file)['hist_mask']

def setup(
    manager: Manager,
    sensor: SPADSensorConfig,
    dashboard: SPADDashboardConfig | None,
    record: bool = False,
    background: bool = True,
):
    """Configures the manager with sensor and dashboard instances.

    Args:
        manager (Manager): Manager to add sensor and dashboard to.
    """

    _sensor: SPADSensor = SPADSensor.create_from_config(sensor)

    # === Accumulate frames for point cloud and background (assume stationary) === #
    global HIST_MASK, BACKGROUND
    background_dir = Path("logs") / "background" 
    if background:
        prompt_string = "Press Enter to calibrate background..."
        input(prompt_string)

        # accumulate frames for point cloud and background
        data = []
        num_samples = _sensor.unwrapped.config.ranging_frequency_hz * 2
        num_samples = 1000
        for _ in tqdm.tqdm(range(num_samples), leave=False, desc="Accumulating background data"):
            data.append(_sensor.accumulate())

        # compute background 
        hists = np.mean([d[SPADDataType.HISTOGRAM] for d in data], axis=0)
        hists = hists.reshape(-1, hists.shape[-1]) # (num_pixels, num_bins)
        BACKGROUND = np.copy(hists)

        # background statistics
        bg_stats = [np.sum(d[SPADDataType.HISTOGRAM]) for d in data]

        from scipy.stats import norm

        # Assume 'samples' is your 1D NumPy array of data points
        global MEAN, STD
        # MEAN, STD = 
        MEAN, STD = norm.fit(np.array(bg_stats))
        n = plt.hist(bg_stats, bins=100, density=True)
        x = np.linspace(np.min(bg_stats), np.max(bg_stats), 300)
        # plot gaussian with mean and std
        plt.plot(x, norm.pdf(x, MEAN, STD))
        gaussian_plot = norm.pdf(x, MEAN, STD) 
        plt.plot(x, gaussian_plot)
        plt.show()
        print(MEAN, STD)

        
        # compute mask to remove 1B signal
        pulse_half_width = 7

        HIST_MASK = np.ones_like(hists.reshape(-1, hists.shape[-1]))
        for i in range(HIST_MASK.shape[0]):
            if BIN_0[i] + pulse_half_width > START_FRAME-1:
                HIST_MASK[i, :BIN_0[i]+pulse_half_width-START_FRAME-1] = 0       

        # save calibration data
        if not background_file.exists():
            background_dir.mkdir(exist_ok=True, parents=True)
            np.savez(background_dir / "calibration.npz", hist_mask=HIST_MASK, background=BACKGROUND)

    input("Press Enter to start data capture...")

    manager.add(sensor=_sensor)

    if dashboard is not None:
        dashboard: SPADDashboard = dashboard.create_from_registry(
            config=dashboard, sensor=_sensor
        )
        dashboard.setup()
        manager.add(dashboard=dashboard)

    #  === Initialize particle filter algorithm === #
    canon_point = '/Users/sidsoma/Desktop/Code/papers/siggraph-24/data/simulated/point.npy'  

    canon_config = CanonConfig(
        num_x                = 400,
        num_y                = 400,
        x_min                = -4,
        x_max                = 4,
        y_min                = -4,
        y_max                = 4,
        num_lct_bins         = 128,
        load_voxel_from_file = False,
        canon_dirs           = [canon_point], # list of canonical directories
    )
    
    particle_filter_config = ParticleFilterConfig(
        x_range              = [-0.6, 0.6],
        y_range              = [-0.6, 0.6],
        z_range              = [0, 3],
        num_particles        = 1000,
        eta                  = 1, # scores will be computed as scores = scores ** eta
        radius               = 0.1, # radius of motion model
        score_fn             = 'dot_product_score', # score function to use
        resampling_fn        = 'residual', # resampling function to use
        sigma                = 0.05, # sigma of KDE
        num_x                = 30, # number of x voxels for kde
        num_y                = 30, # number of y voxels for kde
        num_z                = 30, # number of z voxels for kde
        canon                = canon_config,
    )

    particle_filter_algorithm = ParticleFilterAlgorithm(
        particle_filter_config, sensor_config=sensor
    )
    manager.add(algorithm=particle_filter_algorithm)

    # === Initialize particle filter dashboard === #
    dashboard_config = ParticleFilterDashboardConfig(
        xlim=particle_filter_config.x_range,
        ylim=particle_filter_config.y_range,
        zlim=particle_filter_config.z_range,
        xres=particle_filter_algorithm.xres,
        yres=particle_filter_algorithm.yres,
        zres=particle_filter_algorithm.zres,
        num_x=particle_filter_config.num_x,
        num_y=particle_filter_config.num_y,
        num_z=particle_filter_config.num_z,
        cam_z=CAM_Z,
    )

    dashboard = ParticleFilterDashboard(dashboard_config)
    manager.add(backprojection_dashboard=dashboard)


def loop(
    frame: int,
    manager: Manager,
    sensor: SPADSensor,
    dashboard: SPADDashboard | None = None,
    writer: PklHandler | None = None,
    algorithm: ParticleFilterAlgorithm | None = None,
    backprojection_dashboard: ParticleFilterDashboard | None = None,
):
    """
    Updates dashboard each frame.

    Parameters:
    -----------
        frame (int): Current frame number.
        manager (Manager): Manager controlling the loop.
        sensor (SPADSensor): Sensor instance (unused here).
        dashboard (SPADDashboard): Dashboard instance to update.
    """
    global t0

    if frame % 10 == 0:
        t1 = time.time()
        fps = 10 / (t1 - t0)
        t0 = time.time()
        get_logger().info(f"Frame: {frame}, FPS: {fps:.2f}")

    data = sensor.accumulate()

    if dashboard is not None:
        dashboard.update(frame, data=data)

    if RECORD:
        # save to npz file in log directory
        hists = data[SPADDataType.HISTOGRAM].reshape(
                -1, data[SPADDataType.HISTOGRAM].shape[-1]
            )
        
        pt_cloud = data[SPADDataType.POINT_CLOUD]
        save_dict = {'pt_cloud': pt_cloud, 'hists': hists}
        np.savez(LOGDIR / f"volume_{frame:06d}.npz", **save_dict)

    # === Update particle using measurements === #
    if algorithm is not None:
        assert SPADDataType.POINT_CLOUD in data
        assert SPADDataType.HISTOGRAM in data

        # extract histogram parameters
        num_lct_bins = algorithm.num_lct_bins
        num_pixels = PT_CLOUD_GLOBAL.shape[0]
        hists = data[SPADDataType.HISTOGRAM].reshape(num_pixels, -1)
        num_bins = hists.shape[1]

        # perform anomaly detection
        frame_energy = np.sum(hists)
        print(frame_energy)

        # subtract background from hist and mask out 1b
        hists = np.maximum(hists - BACKGROUND, 0)
        hists *= HIST_MASK

        # normalize t=0 to be at 1-bounce peak
        process_hist = True
        hists_1b_crop = np.zeros((num_pixels, num_lct_bins))
        if process_hist:
            for i in range(num_pixels):
                if START_FRAME-1 < BIN_0[i]:
                    cropped_val = hists[i, BIN_0[i]-(START_FRAME-1):]
                    arr_len = cropped_val.shape[0]
                    hists_1b_crop[i, :arr_len] = cropped_val
                else: 
                    num_bins_pad = (START_FRAME-1) - BIN_0[i]
                    cropped_val = hists[i, :]                    
                    hists_1b_crop[i, num_bins_pad:num_bins_pad+num_bins] = cropped_val
        else: 
            hists_1b_crop[:, :num_bins] = hists

        hists_1b_crop[:, :15] = 0

        # overwrite point cloud and histogram 
        data = {}
        data[SPADDataType.POINT_CLOUD] = PT_CLOUD_GLOBAL
        data[SPADDataType.HISTOGRAM] = hists_1b_crop 
        data['num_sigma_from_mean'] = abs(frame_energy - MEAN) / STD

        # particle filter update
        particles = algorithm.update(data)

        # === Update dashboard to reflect new particle positions === #
        if backprojection_dashboard is not None:
            mean_est = algorithm.particles.mean(dim=0).reshape(1, 3)
            import torch
            rendered_mean = algorithm.canons[0](pt_cloud=torch.Tensor(PT_CLOUD_GLOBAL), deltas=mean_est).squeeze().numpy()
            backprojection_dashboard.update(
                particles,
                hists_1b_crop[:, :num_bins + (START_FRAME-1)],
                PT_CLOUD_GLOBAL,
                rendered_mean[:, :num_bins + (START_FRAME-1)],
            )

    if writer is not None:
        writer.append({"iter": frame, **data})


@register_cli
def nlos_particle_filter_demo(record: bool = False, background: bool = True):
    """Sets up and runs the SPAD dashboard.

    Args:
        sensor (SPADSensorConfig): Configuration object for the sensor.
        dashboard (SPADDashboardConfig): Configuration object for the dashboard.
    """

    global t0
    t0 = time.time()

    with Manager() as manager:
        manager.run(
            setup=partial(setup, record=record, sensor=SENSOR, dashboard=DASHBOARD, background=background),
            loop=loop,
        )


if __name__ == "__main__":
    run_cli(nlos_particle_filter_demo(background=RECAPTURE_BACKGROUND))
