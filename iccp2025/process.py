import numpy as np
from typing import List

def fit_plane_to_pt_clouds(points : np.array) -> np.array:
    """
    Align a set of 3D points to z=0 by rotating about the origin
    and then translating about the new x axis. After transformation,
    (0, 0, 0) should become (0, 0, z_0), for some z_0. 

    Parameters:
    -----------
    points : array containing 3D point cloud (num_points, 3)

    Returns:
    --------
    aligned_points : points aligned to the z=0 plane (num_points, 3)
    """
    # === Compute surface normal of plane (z-axis in new coordinates) === #
    # construct matrix A from points
    A = points - np.mean(points, axis=0, keepdims=True)

    # Compute PCA using SVD 
    _, _, Vt = np.linalg.svd(A)

    # Last PC corresponds to the normal vector of the plane
    normal = Vt[-1] 
    normal = normal / np.linalg.norm(normal)

    if normal[2] > 0:
        normal = -normal

    # === Construct rotation to align points to z-axis === #
    # rotation matrix constructed from elevation angle (phi) and azimuthal angle (theta).
    # coordinate system is aligned so x is left, y is up, z is out.
    # theta and phi are computed to rotate previous normal vector to align to (0, 0, -1)
    #
    from numpy import cos, sin, arcsin, arctan2
    phi = -arcsin(normal[1])              # elevation in y [-pi/2, pi/2]
    theta = np.pi-arctan2(normal[0], normal[2]) # azimuth in x-z [-pi, pi]

    R_y = np.array([[ cos(theta), 0, sin(theta)],
                    [    0      , 1,    0    ],
                    [-sin(theta), 0, cos(theta)]])
    
    R_x = np.array([[ 1,    0    ,     0    ],
                    [ 0, cos(phi), -sin(phi)],
                    [ 0, sin(phi), cos(phi) ]])
    
    R = R_x @ R_y
    
    # === Transform points === #
    aligned_points = points - np.mean(points, axis=0, keepdims=True)
    aligned_points = R @ aligned_points.T # apply transformation (3, num_points)
    aligned_points = aligned_points.T # convert back to inhomogeneous coordinates

    # === Shift points so that camera origin is at (0, 0, -z_0) === #
    cam_position_trans = R @ (np.array([0, 0, 0]) - np.mean(points, axis=0))
    aligned_points[:, 0] -= cam_position_trans[0]
    aligned_points[:, 1] -= cam_position_trans[1]

    # === Determine z_0 === #
    z0 = cam_position_trans[2]

    return aligned_points, z0

def remove_1b(hists : np.array, gates : List[int]) -> np.array:
    """
    Process histograms to remove one bounce peak and normalize t=0.

    Parameters:
    -----------
    hists : histograms (num_pixels, num_bins)

    Returns:
    --------
    hists_1b_crop : histograms (num_pixels, num_bins)
    """

    num_pixels, num_bins = hists.shape    
    glass_gate, start_gate, end_gate = gates


    # === Remove glass gate === #
    hists[..., :glass_gate] = 0

    # === Remove first bounce signal === #
    hist_crop = np.zeros((num_pixels, num_bins))
    for i in range(num_pixels):
        bin0 = np.argmax(hists[i, :]) - 1
        bin_end = num_bins - bin0 
        hist_crop[i, :bin_end] = hists[i, bin0:]

    return hist_crop