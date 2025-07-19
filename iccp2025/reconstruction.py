from copy import deepcopy
import time

import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import importlib
import os
from scipy.sparse import csr_matrix

from cc_hardware.drivers.spads import SPADDataType, SPADSensorConfig
from cc_hardware.utils import Component, config_wrapper
from cc_hardware.utils.constants import C

from process import fit_plane_to_pt_clouds, remove_1b

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from canon import FastSumOfParabolas
from abc import ABC, abstractmethod
from typing import List

"""
Particle Filtering Classes
"""
@config_wrapper
class CanonConfig:
    num_x: int  # number of x voxels for canonical
    num_y: int  # number of y voxels for canonical
    x_min: float  # minimum x value for canonical
    x_max: float  # maximum x value for canonical
    y_min: float  # minimum y value for canonical
    y_max: float  # maximum y value for canonical
    num_lct_bins: int  # number of bins for LCT
    canon_dirs: List[str]  # list of canonical directories
    load_voxel_from_file: bool = True  # whether to load voxelized canonical from file
    t_res : float = -1 # timing resolution of sensor

@config_wrapper
class ParticleFilterConfig:
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    z_range: tuple[float, float]
    num_particles : int
    eta : float # scores will be computed as scores = scores ** eta
    radius : float # radius of motion model
    score_fn : str # score function to use
    resampling_fn : str # resampling function to use
    sigma : float # sigma of KDE
    num_x : int # number of x voxels for kde
    num_y : int # number of y voxels for kde
    num_z : int # number of z voxels for kde
    canon : CanonConfig



class ParticleFilterAlgorithm:
    def __init__(self, config: ParticleFilterConfig, sensor_config: SPADSensorConfig):
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = "cpu"
        self.config = config
        self.sensor_config = sensor_config        
        self.frame = 0
        
        self.canon_config = config.canon
        self.canon_config.t_res = self.sensor_config.timing_resolution 
        self.canons = self._create_canons(self.canon_config) # (num_objects, num_y, num_x, num_bins)

        self.num_particles = config.num_particles
        self.num_objects = len(config.canon.canon_dirs)
        self.radius = config.radius # radius of motion model
        
        self.particles, self.velocity = self._create_particles() # (num_particles, 3)
        self.particles_gpu = self.particles.to('mps', non_blocking=True)

        self.motion_model = RandomWalk(self.num_particles, self.radius) # motion model

        module = importlib.import_module('score') # score function
        self.score_function = getattr(module, config.score_fn)
        self.eta = config.eta # scores will be computed as scores = scores ** eta

        if config.resampling_fn in globals(): # resampling function
            self.resampling_function = globals()[config.resampling_fn]
        else:
            raise ValueError(f"Resampling function '{resampling_fn}' not found!")

        # === KDE parameters === #
        self.sigma = config.sigma # bandwidth for KDE
        num_x = config.num_x; num_y = config.num_y; num_z = config.num_z
        x_vals = torch.linspace(config.x_range[0], config.x_range[1], num_x)
        y_vals = torch.linspace(config.y_range[0], config.y_range[1], num_y)
        z_vals = torch.linspace(config.z_range[0], config.z_range[1], num_z)

        x_grid, y_grid, z_grid = torch.meshgrid(x_vals, y_vals, z_vals)
        # self.volume = torch.stack([x_grid, y_grid, z_grid], dim=-1).to(self.device)
        self.volume = torch.stack([x_grid, y_grid, z_grid], dim=-1).to('mps')

        self.kde_volume = torch.zeros((num_x, num_y, num_z), device='cpu')#.pin_memory()
        self.kde_volume_np = np.zeros((num_x, num_y, num_z))

        # === LCT parameters === #
        self.num_lct_bins = config.canon.num_lct_bins
        self.mtx, self.mtxi = resampling_operator(self.num_lct_bins)
        # self.mtx = torch.Tensor(self.mtx.toarray()).to(self.device)
        # self.mtxi = torch.Tensor(self.mtxi.toarray()).to(self.device)

    def _create_particles(self):
        particles = torch.zeros((self.num_particles, 3 * self.num_objects)).to(self.device)
        velocity = torch.zeros((self.num_particles, 3 * self.num_objects)).to(self.device)

        box_size = [self.config.x_range[1] - self.config.x_range[0], 
                    self.config.y_range[1] - self.config.y_range[0], 
                    self.config.z_range[1] - self.config.z_range[0]]

        box_center = [self.config.x_range[0] + box_size[0] / 2, 
                      self.config.y_range[0] + box_size[1] / 2, 
                      self.config.z_range[0] + box_size[2] / 2]
        
        for i in range(self.num_objects):     
            particles[:, 3*i:3*(i+1)] = random_points_in_box(num_points=self.num_particles, 
                                                             box_width=box_size, 
                                                             center=box_center).to(self.device)
            
            velocity[:, 3*i:3*(i+1)] = random_points_in_sphere(self.num_particles, 
                                                               self.radius).to(self.device)


        # particles *= 0
        # particles[:, 2] = 0.2
        return particles, velocity


    def _create_canons(self, canon_config: CanonConfig):

        canon_dirs = canon_config.canon_dirs
        canon_reps = []
        for i, canon_dir in enumerate(canon_dirs): 
            print(f"Loading canonical {i+1} from {canon_dir}")
            points = np.load(canon_dir) # load point cloud

            # === Load voxelized canonical if exists === #
            if canon_config.load_voxel_from_file and os.path.exists(canon_dir.replace('.npy', '_voxel.npy')):
                loaded_voxel = np.load(canon_dir.replace('.npy', '_voxel.npy')) # load voxelized canonical
            else:
                loaded_voxel = None

            # === Instantiate canonical measurement === #
            canon_rep = FastSumOfParabolas(canon_config, 
                                           points, 
                                           self.device, 
                                           loaded_voxel=loaded_voxel) # instantiate canonical measurement
            
            # === Save voxelized canonical if it doesn't exist === #
            if canon_config.load_voxel_from_file and loaded_voxel is None:
                np.save(canon_dir.replace('.npy', '_voxel.npy'), canon_rep.canon_voxel.detach().cpu().numpy())

            canon_reps.append(canon_rep.to(self.device))

        return canon_reps

    def update(self, data: dict[SPADDataType, np.ndarray]) -> np.ndarray:
        """
        Update the particles based on the measurement.

        Parameters:
        -----------
        data : dict[SPADDataType, np.ndarray]
            - SPADDataType.POINT_CLOUD : (num_pixels, 3)
            - SPADDataType.HISTOGRAM : (num_pixels, numBins)

        Returns:
        --------
        particles : (num_particles, 3)
        """
        assert SPADDataType.POINT_CLOUD in data, "Point cloud missing"
        assert SPADDataType.HISTOGRAM in data, "Histogram missing"

        self.frame += 1

        pt_cloud = data[SPADDataType.POINT_CLOUD]
        hists = data[SPADDataType.HISTOGRAM]

        # === compute LCT of hists === #
        hists_lct = (self.mtx @ hists.T).T # (num_pixels, num_lct_bins)

        # === Move to torch device === #
        hists_np = np.array(hists_lct, dtype=np.float32)  # (num_pixels, num_lct_bins)
        pt_cloud_np = np.array(pt_cloud, dtype=np.float32)# (num_pixels, 3)

        hists = torch.from_numpy(hists_np).to(self.device, non_blocking=True)
        pt_cloud = torch.from_numpy(pt_cloud_np).to(self.device, non_blocking=True)

        # === Update particles and volume === #
        scores = self._evaluate_particles(pt_cloud, hists)
        self._resample_particles(scores)
        cur_particles = self.particles.detach().cpu().numpy()
        self._propagate_particles()
        
        # volume = self._convert_particles_to_volume()
        

        return cur_particles

    def _evaluate_particles(self, pt_cloud: torch.Tensor, hists: torch.Tensor) -> torch.Tensor:
        """
        Compute a score for each particle based on the measurement.

        Parameters:
        -----------
        pt_cloud : point cloud in world coordinates (n_points, 3)
        hists    : space-time histogram (n_points, num_bins)

        Returns:
        --------
        scores : (num_particles, )
        """
        particles = self.particles.clone()
        num_objects = self.num_objects  
        num_pixels, num_bins = hists.shape

        # === reshape and normalize reference image === #
        y_gt = hists.clone()

        # === forward pass === # 
        y_hat = torch.zeros((self.num_particles, 
                                     num_pixels, 
                                     num_bins)).to(self.device, non_blocking=True)        
        for j in range(num_objects):
            batch = particles[:, 3*j:3*(j+1)].clone()
            batch[:, 2] = torch.sign(batch[:, 2]) * batch[:, 2] ** 2 # convert z to v space
            with torch.no_grad():
                cur_render = self.canons[j](pt_cloud=pt_cloud, 
                                            deltas=batch) # (num_particles, num_pixels, num_bins)
                y_hat += cur_render 

        # === compute similarity between images using score function === #
        scores = self.score_function(y_gt.unsqueeze(0).expand(self.num_particles, -1, -1), y_hat)
        # print('scores', scores)
        # print('y_gt', y_gt)
        # print('y_hat', y_hat)
        # print('batch', batch)
        # print(torch.sum(y_hat))

        # plt.figure(figsize=(10, 5))
        # num_rows = 2
        # num_cols = 3
        # plt.subplot(num_rows, num_cols, 1)
        # plt.imshow(y_gt[:, :], cmap='hot')

        # plt.subplot(num_rows, num_cols, 2)
        # scores_test = torch.where(torch.isnan(scores), 0, scores)
        # max_idx = torch.argmax(scores_test)
        # plt.imshow(y_hat[max_idx, :, :], cmap='hot')

        # from matplotlib.colors import LinearSegmentedColormap
        # colors = [(0, 1, 0, 0), (0, 1, 0, 1)]  # Black to Green
        # cmap = LinearSegmentedColormap.from_list("black_to_green", colors)
        # plt.subplot(num_rows, num_cols, 3)
        # plt.imshow(y_gt, cmap='hot')
        # plt.imshow(y_hat[max_idx], cmap=cmap)


        # plt.savefig(f'debug/frame_{self.frame}.png')

        # print('t_res', self.sensor_config.timing_resolution)

        # plt.figure(figsize=(10, 5))
        # num_rows = 4
        # plt.subplot(1, num_rows, 1)
        # plt.plot(pt_cloud[:, 0], pt_cloud[:, 1], 'r.')
        # plt.subplot(1, num_rows, 2)
        # plt.plot(pt_cloud[:, 0], pt_cloud[:, 2], 'r.')
        # plt.subplot(1, num_rows, 3)
        # plt.imshow(y_gt[:, :30], cmap='hot')
        # plt.subplot(1, num_rows, 4)
        # plt.imshow(y_hat[0, :, :30], cmap='hot')
        # plt.show()

        # === Ensure scores are non-negative and remove nan entries === #
        scores_cpu = scores.detach().to('cpu', non_blocking=True)
        nan_idxs = torch.isnan(scores_cpu)
        non_nan_scores = scores_cpu[~nan_idxs]

        min_score = torch.min(non_nan_scores)
        scores_cpu -= min_score

        scores_cpu[nan_idxs] = 0
        scores = scores.to(self.device, non_blocking=True)

        # print(scores)

        return scores

    def _resample_particles(self, scores: torch.Tensor) -> None:
        """
        Resample particles to focus on high-likelihood particles.

        Parameters:
        -----------
        scores : score of each particle (num_particles, )

        Returns:
        --------
        None

        """

        # === Resample particles based on scores === #
        indices = self.resampling_function(scores)

        # === Update particles === #
        self.particles = self.particles[indices]
        self.velocity = self.velocity[indices]

        self.particles_gpu = self.particles_gpu[indices]

    def _propagate_particles(self) -> None:
        """
        Propagate particles based on motion model.
        """
        # === add noise to velocity based on motion model === #
        self.velocity = self.motion_model.forward(velocity=self.velocity)

        # === propagate particle positions === #
        self.particles = self.particles + self.velocity
        
        for i in range(self.num_objects):
            self.particles[:, 3*i+2] = self.particles[:, 3*i+2] * (self.particles[:, 3*i+2] > 0) # restrict to positive z space

    def _convert_particles_to_volume(self) -> np.ndarray:
        """
        Convert particles to a volume.
        """
        mean_est = torch.mean(self.particles, dim=0).numpy()
        volumes = []
        for i in range(self.num_objects):
            idx = [int((mean_est[3*i] - self.config.x_range[0]) / self.xres), 
                   int((mean_est[3*i+1] - self.config.y_range[0]) / self.yres), 
                   int((mean_est[3*i+2] - self.config.z_range[0]) / self.zres)]
            volume = np.zeros((self.config.num_x, self.config.num_y, self.config.num_z))
            volume[idx[0], idx[1], idx[2]] = 1
            volumes.append(volume)

        return volumes


    def _compute_kde(self) -> torch.Tensor:
        """
        Compute the kernel density estimate of the particle distribution.

        Returns:
        --------
        pdfs : list of length num_objects containing (num_x, num_y, num_z) 
                    tensor of the KDE
        """
        num_x, num_y, num_z = self.volume.shape[0:3]

        particles_gpu = self.particles_gpu

        pdfs = []
        for j in range(self.num_objects):            
            # === Get positions for jth object === #
            particles = particles_gpu[:, 3*j:3*(j+1)]
            
            # === Compute KDE of jth object === #
            dists = torch.linalg.norm(self.volume.unsqueeze(0) - particles.unsqueeze(1).unsqueeze(1).unsqueeze(1), dim=-1)
            pdf = torch.exp(-dists**2 / (2 * self.sigma **2))
            pdf = pdf.mean(dim=0)
            
            
            # pdf_cpu = pdf.detach().to('cpu')
            # pdf_np = pdf_cpu.numpy()
            self.kde_volume.copy_(pdf)

            
            np.copyto(self.kde_volume_np, self.kde_volume.numpy())
            pdfs.append(self.kde_volume_np)
            

        return pdfs

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


"""
Resampling Techniques
"""

def systematic(scores : torch.Tensor) -> torch.Tensor:
        """
        Perform systematic resampling on particles based on their scores.
        
        Parameters:
        -----------
        scores    : tensor of particle scores/weights (N, )
        
        Returns:
        --------
        indices   : Resampled particles with shape (N, 3)
        """

        num_particles = scores.shape[0]
        
        # === Normalize scores to probabilities === #
        probabilities = scores / scores.sum()
        
        # === Calculate cumulative sum of probabilities === #
        cumulative_sum = torch.cumsum(probabilities, dim=0)
        
        # === Generate a random starting point === #
        u = torch.rand(1) / num_particles
        
        # === Generate sample points === #
        sample_points = (torch.arange(num_particles, dtype=torch.float32) + u) / num_particles
        
        # === Find indices of particles to be resampled === #
        indices = torch.searchsorted(cumulative_sum, sample_points)
        
        # === Ensure indices are within bounds === #
        indices = torch.clamp(indices, 0, num_particles-1)
        
        return indices

def stratified(scores : torch.Tensor) -> torch.Tensor:
    raise NotImplementedError


def residual(scores : torch.Tensor) -> torch.Tensor:
    """
    Perform residual resampling on a set of particles based on their weights.
    
    Parameters:
    -----------
    scores  : tensor of particle scores (N, )
    
    Returns:
    --------
    indices: Indices of resampled particles (N, )
    """
    N = scores.shape[0]  # Number of particles

    # === Normalize scores to probabilities === #
    probabilities = (scores / scores.sum()).cpu()
    # === Replace NaN values with 0 === #
    probabilities[torch.isnan(probabilities)] = 0
    
    # === Step 1: Compute the deterministic part (integer copies) === #
    num_copies = np.floor(probabilities.numpy() * N).astype(int)  # Integer copies for each particle
    residual = probabilities * N - num_copies # Residual weights
    residual[torch.isnan(residual) | (residual < 0)] = 0 # Replace NaN values with 0
    residual /= residual.sum()                     # Normalize residual weights
    # residual[torch.isnan(residual)] = 0 # Replace NaN values with 0

    # === Add deterministic copies to the indices list === #
    indices = []
    for i in range(N):
        if num_copies[i] != 0:
            indices.extend([i] * num_copies[i])

    # === Step 2: Redistribute remaining particles using multinomial sampling === #
    num_residual_particles = N - len(indices) # Remaining particles to sample
    if num_residual_particles > 0:
        residual_indices = np.random.choice(range(N), size=num_residual_particles, p=residual)
        indices.extend(residual_indices)
    
    return torch.Tensor(indices).long().to(scores.device)


def multinomial(scores : torch.Tensor, thresh_pct : float = 0.2) -> torch.Tensor:
    """
    Sample indices from multinomial distribution.

    Parameters:
    -----------
    scores : tensor of particle scores (num_particles, )

    Returns:
    --------
    indices : indices of resampled particles (num_particles, )
    """

    num_particles = scores.shape[0]

    # === Clip scores to be greater than 0 === #
    # scores = torch.clip(scores, min=0)        

    # === Only use top k particles === #
    k = int(num_particles * thresh_pct)
    scores = top_k_mask(scores, k=k) * scores

    # === Normalize scores to get probabilities === #
    probabilities = scores / torch.sum(scores)
    # probabilities = torch.exp(probabilities * (1/5) * scores) 
    # probabilities = probabilities / torch.sum(probabilities) # normalize again
    
    # === Resample particles based on their probabilities === #
    dist = torch.distributions.categorical.Categorical(probs=probabilities)
    indices = dist.sample((num_particles, ))

    return indices


        
""" 
Motion Model Parametrizations 
"""

class MotionModel(ABC):
    def __init__(self, num_particles):
        super().__init__()
        self.num_particles = num_particles

    def forward(self, velocity: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        -----------
        velocity : velocity of particle in previous frame (num_points, 3*num_objects)

        Returns:
        --------
        dx : new velocities (num_points, 3*num_objects)
        """
        pass

class RandomWalk(MotionModel):  
    def __init__(self, num_particles: int, radius: float):
        super().__init__(num_particles)

        self.radius = radius

    def forward(self, velocity: torch.Tensor) -> torch.Tensor:
        """
        Simulates displacements resulting from a random walk in 3D.
        Velocity isn't used here because previous velocity doesn't affect
        future velocities in a random walk.
        """
        
        # === Create zero-mean 3D Gaussian with stddev = radius === #
        dx = torch.randn_like(velocity).to(velocity.device) * self.radius
        # dx = torch.randn_like(velocity).to(velocity.device) 
        # dx /= torch.norm(dx, dim=-1, keepdim=True) 
        # dx *= self.radius

        return dx

class ConstantVelocity(MotionModel):
    def __init__(self, num_particles, radius : float, **kwargs):
        super().__init__(num_particles)

        self.radius = radius


    def forward(self, velocity: torch.Tensor):
        """
        Adds Gaussian noise to velocities. The mean of the Gaussian
        is the previous velocity, variance is 0.1 m. 
        """
        assert velocity.shape[1] % 3 == 0, "Velocity must be of shape (num_particles, 3*num_objects)"
        num_objects = velocity.shape[1] // 3

        # === Variance of motion model === #
        variance = torch.Tensor([self.radius]).to(velocity.device)

        # === Mean velocity of all particles === #
        mean_velocity = torch.mean(velocity, axis=0).reshape(1, 3*num_objects)
 
        # === Zero-mean, constant variance Gaussian noise === #
        eps = torch.randn_like(velocity).to(velocity.device) * variance #torch.sqrt(variance)

        # === Add noise to velocity === #
        new_velocity = velocity + eps
        # new_velocity = mean_velocity + eps

        return new_velocity

"""
Helper Functions
"""

def random_points_in_box(num_points: int, 
                         box_width: List[float], 
                         center: List[float]
            ) -> torch.Tensor:
    """
    Samples random points within a 3D cube as an initialization for 
    particle positions.

    Parameters:
    -----------
    num_points  : number of particles
    box_width   : width of initialized region
    center      : center of initialized region 

    Returns:
    --------
    points   : initialized particle locations (num_points, 3)
    """

    x = (torch.rand(num_points) - 0.5) * box_width[0] + center[0]
    y = (torch.rand(num_points) - 0.5) * box_width[1] + center[1]
    z = (torch.rand(num_points) - 0.5) * box_width[2] + center[2]

    points = torch.stack([x, y, z], dim=-1) 
    
    return points

def random_points_in_sphere(num_points: int, 
                            radius: float
                    ) -> torch.Tensor:
    """
    Samples random points within a 3D sphere as an initialization for 
    particle positions.

    Parameters:
    -----------
    num_points : number of particles
    radius     : max radius of sphere

    Returns:
    --------
    points   : initialized particle locations (num_points, 3)
    """
    # === randomly choose radius in the range [0, r] === #
    r = radius * torch.pow(torch.rand(num_points), 1/3)

    # === uniformly distribute theta (azimuthal angle) in [0, 2*pi] === #
    phi = 2 * torch.pi * torch.rand(num_points)

    # === uniformly distribute phi (polar angle) === #
    theta = torch.pi * torch.rand(num_points)

    # === convert spherical to cartesian coordinates === #
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta) 

    # === concatneate coordinates === #
    points = torch.stack([x, y, z], dim=1)

    return points


def top_k_mask(vector, k=20):    
    if k >= vector.shape[0]:
        return torch.ones_like(vector).to(vector.device)

    # === Get the indices of the top k highest entries in the vector === #
    top_k_values, top_k_indices = torch.topk(vector, k)
    
    # Create a boolean mask with the same shape as the vector
    mask = torch.zeros_like(vector).to(vector.device)
    
    # Set the positions of the top k entries to True in the mask
    mask[top_k_indices] = 1
    
    return mask

def resampling_operator(numBins):
    """
    Function adapted from O'Toole et al. "Confocal NLOS Imaging using Light Cone Transform".
    
    Parameters:
    -----------
    numBins : number of bins in original histogram 
    
    Returns:
    --------
    mtx  : matrix mapping from native -> LCT space (newNumBins, newNumBins)
    mtxi : inverse mapping from LCT space -> native (newNumBins, newNumBins)

    """
    mtx = csr_matrix(([], ([], [])), shape=(numBins**2, numBins))

    x = np.arange(1, numBins**2 + 1)
    mtx[x - 1, np.ceil(np.sqrt(x)) - 1] = 1

    # mtx  = spdiags(1./sqrt(x)', 0, M**2, M**2) * mtx
    x_sqrt_inv = 1 / np.sqrt(x)
    diag_mtx = csr_matrix((x_sqrt_inv, (np.arange(numBins**2), np.arange(numBins**2))), shape=(numBins**2, numBins**2))
    mtx = diag_mtx.dot(mtx)

    mtxi = mtx.transpose()

    K = np.round(np.log(numBins) / np.log(2))
    for k in range(int(K)):
        mtx = 0.5 * (mtx[0::2, :] + mtx[1::2, :])
        mtxi = 0.5 * (mtxi[:, 0::2] + mtxi[:, 1::2])

    return mtx, mtxi
