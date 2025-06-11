import time
from functools import partial
from pathlib import Path
import threading
import multiprocessing

import torch
import numpy as np

from cc_hardware.tools.dashboard import Dashboard
from cc_hardware.drivers.spads import SPADDataType, SPADSensor, SPADSensorConfig
from cc_hardware.drivers.spads.pkl import PklSPADSensorConfig
from cc_hardware.drivers.stepper_motors import StepperMotorSystem
from cc_hardware.drivers.stepper_motors.stepper_controller import (
    SnakeControllerAxisConfig,
    SnakeStepperControllerConfigXY,
    StepperController,
)
from cc_hardware.drivers.spads.spad_wrappers import SPADMovingAverageWrapperConfig
from cc_hardware.drivers.stepper_motors.telemetrix_stepper import (
    SingleDrive1AxisGantryConfig,
)
from cc_hardware.utils import (
    AtomicVariable,
    MPAtomicVariable,
    Manager,
    ThreadedComponent,
    get_logger,
    register_cli,
    run_cli,
    threaded_component,
)
from cc_hardware.utils.constants import TORCH_DEVICE
from cc_hardware.utils.file_handlers import PklReader

from gui.simple_gui import CVPR25DashboardConfig, CVPR25Dashboard
from gui.full_gui import CVPR25FullDashboardConfig, CVPR25FullDashboard
from gui.mpc.simulation import get_path
from ml.model import DeepLocation8

# ==========

STEPPER_SYSTEM = SingleDrive1AxisGantryConfig.create()
X_RANGE = (0, 32)
Y_RANGE = (8, 36)
STEPPER_CONTROLLER = SnakeStepperControllerConfigXY.create(
    axes=dict(
        x=SnakeControllerAxisConfig(range=X_RANGE, samples=3),
        y=SnakeControllerAxisConfig(range=Y_RANGE, samples=2),
    )
)

GUI = CVPR25DashboardConfig.create(
    x_range=(0, 32),
    y_range=(0, 32),
    point_size=10.0,
)
GUI = CVPR25FullDashboardConfig.create(
    x_range=(0, 32),
    y_range=(0, 32),
    dot_size=15,
)

PATH_THREAD: multiprocessing.Process | None = None
PATH_THREAD_EVENT = multiprocessing.Event()

# ==========

POSITIONS = []
LAST_POSITION = None
CONTROLLER_POS = AtomicVariable((0.0, 0.0))
PREDICTED_POS: MPAtomicVariable
PATH: MPAtomicVariable 
HISTOGRAM = AtomicVariable(None)

# ==========


def stepper_callback(
    future,
    *,
    manager: Manager,
    stepper_system: StepperMotorSystem | ThreadedComponent,
    controller: StepperController,
    i: int,
    repeat: bool = True,
):
    if not manager.is_looping:
        get_logger().info("Manager is not looping, stopping stepper callback.")
        return

    if repeat:
        i %= controller.total_positions

    pos = controller.get_position(i)
    CONTROLLER_POS.set((pos["x"], pos["y"]))
    stepper_system.move_to(pos["x"], pos["y"]).add_done_callback(
        partial(
            stepper_callback,
            manager=manager,
            stepper_system=stepper_system,
            controller=controller,
            i=i + 1,
        )
    )


def sensor_callback(
    future,
    *,
    manager: Manager,
    sensor: SPADSensor | ThreadedComponent,
):
    if not manager.is_looping:
        get_logger().info("Manager is not looping, stopping stepper callback.")
        return

    data = future.result()
    assert SPADDataType.HISTOGRAM in data, "Sensor must support histogram data type."
    HISTOGRAM.set(data[SPADDataType.HISTOGRAM])

    sensor.accumulate().add_done_callback(
        partial(
            sensor_callback,
            manager=manager,
            sensor=sensor,
        )
    )


def path_callback(x_range, y_range, PREDICTED_POS: MPAtomicVariable, PATH: MPAtomicVariable):
    from gui.mpc.map import Map, Obstacle
    from gui.mpc.MPC import MPC
    from gui.mpc.reference_path import ReferencePath
    from gui.mpc.spatial_bicycle_models import BicycleModel
    from scipy import sparse

    # Load map file
    map = Map(file_path="data/maps/sim_map.png", origin=[-1, -2], resolution=0.005)

    # Specify waypoints
    wp_x = [1.25, -0.75, -0.75]
    wp_y = [0, 0, -1.5]

    # Specify path resolution
    path_resolution = 0.05  # m / wp

    # Create smoothed reference path
    reference_path = ReferencePath(
        map,
        wp_x,
        wp_y,
        path_resolution,
        smoothing_distance=5,
        max_width=0.4,
        circular=False,
    )

    # Instantiate motion model
    car = BicycleModel(
        length=0.012, width=0.006, reference_path=reference_path, Ts=0.05
    )

    ##############
    # Controller #
    ##############

    N = 55
    Q = sparse.diags([1.0, 0.0, 0.0])
    R = sparse.diags([0.5, 0.0])
    QN = sparse.diags([1.0, 0.0, 0.0])

    v_max = 1.0  # m/s
    delta_max = 0.66  # rad
    ay_max = 4.0  # m/s^2
    InputConstraints = {
        "umin": np.array([0.0, -np.tan(delta_max) / car.length]),
        "umax": np.array([v_max, np.tan(delta_max) / car.length]),
    }
    StateConstraints = {
        "xmin": np.array([-np.inf, -np.inf, -np.inf]),
        "xmax": np.array([np.inf, np.inf, np.inf]),
    }
    mpc = MPC(car, N, Q, R, QN, StateConstraints, InputConstraints, ay_max)

    # Compute speed profile
    a_min = -0.1  # m/s^2
    a_max = 0.5  # m/s^2
    SpeedProfileConstraints = {
        "a_min": a_min,
        "a_max": a_max,
        "v_min": 0.0,
        "v_max": v_max,
        "ay_max": ay_max,
    }
    car.reference_path.compute_speed_profile(SpeedProfileConstraints)
    while not PATH_THREAD_EVENT.is_set():
        pos = PREDICTED_POS.get()
        if pos is not None:
            x, y = pos

            # Add obstacles
            cx_n = np.interp(y, x_range, [-0.65, -0.85])
            cy_n = np.interp(x, y_range, [-0.2, -0.4])

            map.remove_obstacles(map.obstacles)
            obs = Obstacle(cx=cx_n, cy=cy_n, radius=0.1)
            map.add_obstacles([obs])

            # Until arrival at end of path
            car = BicycleModel(
                length=0.012, width=0.006, reference_path=reference_path, Ts=0.05
            )
            mpc.model = car
            while car.s < reference_path.length:
                # Get control signals
                try:
                    u = mpc.get_control()
                except IndexError:
                    break

                # Simulate car
                car.drive(u)

            x_n, y_n = mpc.current_prediction

            v = 200
            x_w = np.interp(x_n, [-1, 2], [-v, v]) + 70
            y_w = np.interp(y_n, [-1, 2], [-v, v]) + 110
            path_xyz = np.c_[y_w, x_w, np.zeros_like(x_w)]

            PATH.set(path_xyz)


# ==========


def create_sensor(
    manager: Manager, config: dict, sensor_port: str | None = None
) -> tuple[SPADSensor | ThreadedComponent, SPADSensorConfig]:
    assert "sensor" in config, "Configuration must contain 'sensor' key."
    sensor_config: SPADSensorConfig = config["sensor"]
    sensor = SPADMovingAverageWrapperConfig.create(
        wrapped=sensor_config,
        window_size=10,
    )
    sensor = SPADSensor.create_from_config(sensor, port=sensor_port)
    sensor = threaded_component(sensor)
    sensor.accumulate().add_done_callback(
        partial(
            sensor_callback,
            manager=manager,
            sensor=sensor,
        )
    )
    manager.add(sensor=sensor)
    return sensor, sensor_config


def create_stepper_system(
    manager: Manager, stepper_port: str | None = None
) -> tuple[StepperMotorSystem | ThreadedComponent, StepperController]:
    if stepper_port is not None:
        STEPPER_SYSTEM.port = stepper_port
    controller = StepperController.create_from_config(STEPPER_CONTROLLER)

    stepper_system = StepperMotorSystem.create_from_config(STEPPER_SYSTEM)
    stepper_system = threaded_component(stepper_system)
    stepper_system.initialize().result()

    stepper_system.move_to(0, 0).add_done_callback(
        partial(
            stepper_callback,
            manager=manager,
            stepper_system=stepper_system,
            controller=controller,
            i=0,
        )
    )

    manager.add(stepper_system=stepper_system, controller=controller)

    return stepper_system, controller


def load_model(
    manager: Manager, model_path: Path, sensor_config: SPADSensorConfig
) -> DeepLocation8:
    assert model_path.exists(), f"Model file {model_path} does not exist."
    model = DeepLocation8(
        sensor_config.height, sensor_config.width, sensor_config.num_bins
    ).to(TORCH_DEVICE)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(TORCH_DEVICE)

    manager.add(model=model)
    return model


def create_gui(manager: Manager) -> Dashboard:
    if isinstance(GUI, CVPR25FullDashboardConfig):
        gui = CVPR25FullDashboard(GUI)
    elif isinstance(GUI, CVPR25DashboardConfig):
        gui = CVPR25Dashboard(GUI)
    gui.setup()
    manager.add(gui=gui)

    # Call path callback in thread
    global PATH_THREAD
    if PATH_THREAD is None or not PATH_THREAD.is_alive():
        PATH_THREAD = multiprocessing.Process(
            target=partial(
                path_callback,
                x_range=gui.config.x_range,
                y_range=gui.config.y_range,
                PREDICTED_POS=PREDICTED_POS,
                PATH=PATH,
            ),
            daemon=True,
        )
        PATH_THREAD.start()

    return gui


# ==========


def setup(
    manager: Manager,
    config_path: Path,
    model_path: Path,
    sensor_port: str | None = None,
    stepper_port: str | None = None,
):
    """Configures the manager with sensor and dashboard instances.

    Args:
        manager (Manager): Manager to add sensor and dashboard to.
    """
    config = PklReader.load_all(config_path)
    assert len(config) == 1, "Expected exactly one configuration in the pickle file."
    config = config[0]

    _, sensor_config = create_sensor(manager, config, sensor_port=sensor_port)
    load_model(manager, model_path, sensor_config)
    create_gui(manager)
    create_stepper_system(manager, stepper_port=stepper_port)


def loop(
    frame: int,
    model: DeepLocation8,
    gui: Dashboard,
    **kwargs,
):
    global LAST_POSITION

    histogram = HISTOGRAM.get()
    if len(histogram) == 0:
        get_logger().warning("No histogram available for evaluation.")
        return
    positions = model.evaluate(histogram)
    POSITIONS.append(positions[0])
    if len(POSITIONS) > 20:
        POSITIONS.pop(0)

    position = np.mean(POSITIONS, axis=0)
    position = np.array((position[1], position[0]))
    if LAST_POSITION is not None:
        position = LAST_POSITION + 0.1 * (position - LAST_POSITION)
    LAST_POSITION = position
    PREDICTED_POS.set(position)
    gui.update(
        frame=frame,
        positions=[position],
        # gt_positions=[CONTROLLER_POS.get()],
        path=PATH.get(),
    )


def cleanup(
    gui: Dashboard,
    stepper_system: StepperMotorSystem,
    **kwargs,
):
    get_logger().info("Cleaning up...")
    gui.close()

    stepper_system.move_to(0, 0)
    stepper_system.close()

    global PATH_THREAD
    if PATH_THREAD is not None and PATH_THREAD.is_alive():
        PATH_THREAD_EVENT.set()
        PATH_THREAD.join(timeout=1.0)
        if PATH_THREAD.is_alive():
            get_logger().warning("Path thread did not terminate properly.")
        PATH_THREAD = None


@register_cli
def cvpr2025(
    config: Path,
    model: Path,
    sensor_port: str | None = None,
    stepper_port: str | None = None,
):
    """Sets up and runs the SPAD dashboard.

    Args:
        sensor (SPADSensorConfig): Configuration object for the sensor.
        dashboard (SPADDashboardConfig): Configuration object for the dashboard.
    """

    global t0
    t0 = time.time()

    with Manager() as manager:
        manager.run(
            setup=partial(
                setup,
                config_path=config,
                model_path=model,
                sensor_port=sensor_port,
                stepper_port=stepper_port,
            ),
            loop=loop,
            cleanup=cleanup,
        )


if __name__ == "__main__":
    PREDICTED_POS = MPAtomicVariable((0.0, 0.0))
    PREDICTED_POS.get()
    PATH = MPAtomicVariable(None)
    PATH.get()

    run_cli(cvpr2025)
