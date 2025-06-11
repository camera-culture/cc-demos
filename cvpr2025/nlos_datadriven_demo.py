import time
from functools import partial
from pathlib import Path

import torch

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
from ml.model import DeepLocation8

# ==========

STEPPER_SYSTEM = SingleDrive1AxisGantryConfig.create()
STEPPER_CONTROLLER = SnakeStepperControllerConfigXY.create(
    axes=dict(
        x=SnakeControllerAxisConfig(range=(0, 32), samples=3),
        y=SnakeControllerAxisConfig(range=(0, 32), samples=2),
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

# ==========

CONTROLLER_POS = AtomicVariable((0.0, 0.0))
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
    histogram = HISTOGRAM.get()
    if len(histogram) == 0:
        get_logger().warning("No histogram available for evaluation.")
        return
    positions = model.evaluate(histogram)

    gui.update(
        frame=frame,
        positions=positions.tolist(),
        gt_positions=[CONTROLLER_POS.get()],
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
    run_cli(cvpr2025)
