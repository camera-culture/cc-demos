import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

from gui.mpc.map import Map, Obstacle
from gui.mpc.MPC import MPC
from gui.mpc.reference_path import ReferencePath
from gui.mpc.spatial_bicycle_models import BicycleModel


def get_path(cx=-0.75, cy=-0.8, show=False):
    # Load map file
    map = Map(file_path='data/maps/sim_map.png', origin=[-1, -2],
                resolution=0.005)

    # Specify waypoints
    wp_x = [1.25, -0.75, -0.75]
    wp_y = [0, 0, -1.5]

    # Specify path resolution
    path_resolution = 0.05  # m / wp

    # Create smoothed reference path
    reference_path = ReferencePath(map, wp_x, wp_y, path_resolution,
                                    smoothing_distance=5, max_width=0.4,
                                    circular=False)

    # Add obstacles
    obs = Obstacle(cx=cx, cy=cy, radius=0.1)
    map.add_obstacles([obs])

    # Instantiate motion model
    car = BicycleModel(length=0.012, width=0.006,
                        reference_path=reference_path, Ts=0.05)

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
    InputConstraints = {'umin': np.array([0.0, -np.tan(delta_max)/car.length]),
                        'umax': np.array([v_max, np.tan(delta_max)/car.length])}
    StateConstraints = {'xmin': np.array([-np.inf, -np.inf, -np.inf]),
                        'xmax': np.array([np.inf, np.inf, np.inf])}
    mpc = MPC(car, N, Q, R, QN, StateConstraints, InputConstraints, ay_max)

    # Compute speed profile
    a_min = -0.1  # m/s^2
    a_max = 0.5  # m/s^2
    SpeedProfileConstraints = {'a_min': a_min, 'a_max': a_max,
                               'v_min': 0.0, 'v_max': v_max, 'ay_max': ay_max}
    car.reference_path.compute_speed_profile(SpeedProfileConstraints)

    ##############
    # Simulation #
    ##############

    # Until arrival at end of path
    while car.s < reference_path.length:

        # Get control signals
        try:
            u = mpc.get_control()
        except IndexError:
            break

        # Simulate car
        car.drive(u)

        # Plot path and drivable area
        if show:
            reference_path.show()

        # Plot car
        if show:
            car.show()

        # Plot MPC prediction
        if show:
            mpc.show_prediction()

    if show:
        plt.show()

    return mpc.current_prediction

if __name__ == "__main__":
    path = get_path()
    print("Optimal path:", path)