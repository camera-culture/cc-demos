import multiprocessing as mp

import numpy as np
from scipy import sparse

from cc_hardware.utils import Component, config_wrapper, Config

from mpc.map import Map, Obstacle
from mpc.MPC import MPC
from mpc.reference_path import ReferencePath
from mpc.spatial_bicycle_models import BicycleModel

@config_wrapper
class PlannerConfig(Config):
    """
    Configuration for the Planner component.
    """
    x_range: tuple
    y_range: tuple

class Planner(Component[PlannerConfig]):
    def __init__(self, config: PlannerConfig):
        super().__init__(config)

        self._path: np.ndarray = None
        self._pos: np.ndarray = None

        self._process: mp.Process = None
        self._stop_event = mp.Event()
        self._lock = mp.Lock()

    def _start_background_process(self):
        """
        Start the background process for the Planner component.
        """
        self._process = mp.Process(target=self._run_planner)
        self._process.start()

    def _run_planner(self):
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

        while not self._stop_event.is_set():
            pos = self.pos
            if pos is None:
                continue

            x, y = pos

    @property
    def pos(self) -> np.ndarray:
        """
        Get the current position of the Planner component.
        """
        with self._lock:
            return self._pos

    @pos.setter
    def pos(self, value: np.ndarray):
        """
        Set the current position of the Planner component.
        """
        with self._lock:
            self._pos = value

    @property
    def path(self) -> np.ndarray:
        """
        Get the current path of the Planner component.
        """
        with self._lock:
            return self._path