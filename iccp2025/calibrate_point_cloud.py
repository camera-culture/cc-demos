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

global LOG_DIR
LOG_DIR: Path = Path("logs") / "pt_cloud"
LOG_DIR.mkdir(exist_ok=True, parents=True)


SENSOR = WRAPPED_SENSOR
DASHBOARD = PyQtGraphDashboardConfig.create(fullscreen=True)
DASHBOARD = DummySPADDashboardConfig.create()

def setup(
    manager: Manager,
    sensor: SPADSensorConfig,
    dashboard: SPADDashboardConfig | None,
):
    """Configures the manager with sensor and dashboard instances.

    Args:
        manager (Manager): Manager to add sensor and dashboard to.
    """

    _sensor: SPADSensor = SPADSensor.create_from_config(sensor)

    # === Accumulate frames for point cloud and background (assume stationary) === #
    # accumulate frames for point cloud and background
    data = []
    for _ in tqdm.tqdm(range(_sensor.unwrapped.config.ranging_frequency_hz * 2), leave=False, desc="Accumulating background data"):
        data.append(_sensor.accumulate())
    
    # compute point cloud, background, and t0 from average of first N frames
    pt_cloud = np.nanmean([d[SPADDataType.POINT_CLOUD] for d in data], axis=0)
    pt_cloud, cam_z = fit_plane_to_pt_clouds(pt_cloud)

    # swap coordinate system to +x is left
    x_coords = np.copy(pt_cloud[:, 1]) # swap x and y
    y_coords = np.copy(pt_cloud[:, 0]) # swap x and y
    
    pt_cloud[:, 0] = -x_coords # -x points left
    pt_cloud[:, 1] = y_coords    

    # save point cloud data
    cam_z = abs(cam_z)
    np.savez(LOG_DIR / "calibration.npz", pt_cloud=pt_cloud, cam_z=cam_z)

    # visualize point cloud
    plt.figure(figsize=(15, 5))

    plt.title(f"Point Cloud (camera z = {cam_z:.2f} m)")
    
    # remove spines of plot
    for ax in plt.gcf().axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    # remove ticks
    for ax in plt.gcf().axes:
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

    plt.subplot(1, 3, 1)
    plt.scatter(pt_cloud[:, 0], pt_cloud[:, 1])
    plt.subplot(1, 3, 2)
    plt.scatter(pt_cloud[:, 0], pt_cloud[:, 2])
    plt.subplot(1, 3, 3)
    plt.scatter(pt_cloud[:, 1], pt_cloud[:, 2])
    
    plt.savefig(LOG_DIR / "pt_cloud.png")

    input(f"Point cloud saved to {LOG_DIR}. ctrl-c to close program...")


def loop(
    frame: int,
    manager: Manager,
    sensor: SPADSensor,
    dashboard: SPADDashboard | None = None,
    writer: PklHandler | None = None,
    algorithm: ParticleFilterAlgorithm | None = None,
    backprojection_dashboard: ParticleFilterDashboard | None = None,
):
    pass


@register_cli
def save_pt_cloud():
    """Sets up and runs the SPAD dashboard.

    Args:
        sensor (SPADSensorConfig): Configuration object for the sensor.
        dashboard (SPADDashboardConfig): Configuration object for the dashboard.
    """

    global t0
    t0 = time.time()

    with Manager() as manager:
        manager.run(
            setup=partial(setup, 
                          sensor=SENSOR, 
                          dashboard=DASHBOARD),
                          loop=loop,
        )


if __name__ == "__main__":
    run_cli(save_pt_cloud())
