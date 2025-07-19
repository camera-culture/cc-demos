import time
from datetime import datetime
from functools import partial
from pathlib import Path
import numpy as np
import tqdm

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
RECORD = False
CAPTURE_BACKGROUND = True
RECAPTURE_CALIBRATION = False

WRAPPED_SENSOR_CONFIG = VL53L8CHConfig4x4.create(
    num_bins=48,
    subsample=1,
    start_bin=1, # 16
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


SENSOR = WRAPPED_SENSOR

DASHBOARD = PyQtGraphDashboardConfig.create(fullscreen=True)
DASHBOARD = DummySPADDashboardConfig.create()

BACKGROUND = None
PT_CLOUDS = []

if RECORD: 
    LOGDIR.mkdir(exist_ok=True, parents=True)

global START_FRAME 
START_FRAME = WRAPPED_SENSOR_CONFIG.start_bin

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
    global PT_CLOUD_GLOBAL, BIN_0, CAM_Z, HIST_MASK, BACKGROUND
    if RECAPTURE_CALIBRATION:
        if background:
            prompt_string = "Press Enter to calibrate point cloud and background..."
        else:
            prompt_string = "Press Enter to calibrate point cloud..."

        input(prompt_string)

        # accumulate frames for point cloud and background
        data = []
        for _ in tqdm.tqdm(range(_sensor.unwrapped.config.ranging_frequency_hz * 2), leave=False, desc="Accumulating background data"):
            data.append(_sensor.accumulate())
        
        # compute point cloud, background, and t0 from average of first N frames
        PT_CLOUD_GLOBAL = np.nanmean([d[SPADDataType.POINT_CLOUD] for d in data], axis=0)
        PT_CLOUD_GLOBAL, CAM_Z = fit_plane_to_pt_clouds(PT_CLOUD_GLOBAL)

        # swap coordinate system to +x is left
        x_coords = np.copy(PT_CLOUD_GLOBAL[:, 1]) # swap x and y
        y_coords = np.copy(PT_CLOUD_GLOBAL[:, 0]) # swap x and y
        
        PT_CLOUD_GLOBAL[:, 0] = -x_coords # -x points left
        PT_CLOUD_GLOBAL[:, 1] = y_coords

        hists = np.mean([d[SPADDataType.HISTOGRAM] for d in data], axis=0)
        BIN_0 = np.argmax(hists[..., :], axis=-1).reshape(-1) # (num_pixels, )
        
        # compute mask to remove 1B signal
        pulse_half_width = 7

        HIST_MASK = np.ones_like(hists.reshape(-1, hists.shape[-1]))
        for i in range(HIST_MASK.shape[0]):
            HIST_MASK[i, :BIN_0[i]+pulse_half_width] = 0

        if background:
            BACKGROUND = np.mean([d[SPADDataType.HISTOGRAM] for d in data], axis=0)
            BACKGROUND = BACKGROUND.reshape(-1, BACKGROUND.shape[-1])

        # save calibration data
        np.savez("calibration.npz", 
                 pt_cloud=PT_CLOUD_GLOBAL, 
                 bin_0=BIN_0, 
                 cam_z=CAM_Z, 
                 hist_mask=HIST_MASK, 
                 background=BACKGROUND)
    else: 
        calib_data = np.load("calibration.npz")
        PT_CLOUD_GLOBAL = calib_data['pt_cloud']
        BIN_0 = calib_data['bin_0']
        CAM_Z = calib_data['cam_z']
        HIST_MASK = calib_data['hist_mask']
        BACKGROUND = calib_data['background']

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
        num_x                = 200,
        num_y                = 200,
        x_min                = -2,
        x_max                = 2,
        y_min                = -2,
        y_max                = 2,
        num_lct_bins         = 64,
        load_voxel_from_file = False,
        canon_dirs           = [canon_point], # list of canonical directories
    )
    
    particle_filter_config = ParticleFilterConfig(
        x_range              = [-0.5, 0.5],
        y_range              = [-0.5, 0.5],
        z_range              = [0, 1],
        num_particles        = 1000,
        eta                  = 8, # scores will be computed as scores = scores ** eta
        radius               = 0.05, # radius of motion model
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
        cam_z=-CAM_Z,
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

        # process histogram to remove 1-bounce
        num_lct_bins = algorithm.num_lct_bins
        glass_gate, start_gate, end_gate = 0, 0, num_lct_bins
        num_pixels = PT_CLOUD_GLOBAL.shape[0]
        hists = data[SPADDataType.HISTOGRAM].reshape(num_pixels, -1)
        num_bins = hists.shape[1]

        # mask out 1b signal
        hist_mask_offset = np.ones_like(HIST_MASK)
        for i in range(num_pixels):
            hist_mask_offset[i, :num_bins-START_FRAME+1] = HIST_MASK[i, START_FRAME-1:]
    
        hists *= hist_mask_offset

        # normalize t=0 to be at 1-bounce peak
        process_hist = True
        hists_1b_crop = np.zeros((num_pixels, num_lct_bins))
        if process_hist:
            for i in range(num_pixels):
                if START_FRAME-1 < BIN_0[i]:
                    cropped_val = hists[i, BIN_0[i]-START_FRAME+1:]
                else: 
                    cropped_val = hists[i, :]

                if BACKGROUND is not None and False:
                    cropped_val = np.maximum(cropped_val - BACKGROUND[i, BIN_0[i]-START_FRAME+1:], 0)
                
                if START_FRAME-1 < BIN_0[i]:
                    hists_1b_crop[i, :num_bins-BIN_0[i]+START_FRAME-1] = cropped_val
                else:
                    idx1 = START_FRAME-1-BIN_0[i]
                    arr_len = min(num_lct_bins - idx1, cropped_val.shape[0])
                    hists_1b_crop[i, idx1:idx1+arr_len] = cropped_val[:arr_len]
        else: 
            hists_1b_crop[:, :num_bins] = hists

        # overwrite point cloud and histogram 
        data = {}
        data[SPADDataType.POINT_CLOUD] = PT_CLOUD_GLOBAL
        data[SPADDataType.HISTOGRAM] = hists_1b_crop 

        # particle filter update
        particles = algorithm.update(data)

        # === Update dashboard to reflect new particle positions === #
        if backprojection_dashboard is not None:
            backprojection_dashboard.update(
                particles,
                hists_1b_crop,
                PT_CLOUD_GLOBAL,
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
    run_cli(nlos_particle_filter_demo(background=CAPTURE_BACKGROUND))
