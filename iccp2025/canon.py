import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
import os
import time

SPEED_OF_LIGHT = 3E8

class FastSumOfParabolas(nn.Module):
    def __init__(self,
                configs: dict,
                points: np.array,
                device: str,
                loaded_voxel: np.array = None
            ) -> None:
        """
        Class implementation for canonical measurement.

        Parameters:
        -----------
        configs  : config variables
        points   : point cloud of object (num_points, 3)
        device   : gpu/cpu/mps device
        """
        
        super().__init__()

        self.device = device

        # === Define grid for canonical measurement === #
        self.num_x = configs.num_x
        self.num_y = configs.num_y

        x_min = configs.x_min; x_max = configs.x_max
        y_min = configs.y_min; y_max = configs.y_max

        self.x_min = x_min; self.x_max = x_max
        self.y_min = y_min; self.y_max = y_max

        # === Important timing parameters === #
        self.t_res = configs.t_res

        self.numBins = configs.num_lct_bins
        self.v_range = ((SPEED_OF_LIGHT * self.numBins * self.t_res / 2) ** 2)
        self.v_res = self.v_range / self.numBins
       
        # === Determine base v_res and camera v_res === #
        self.num_sub_bins = 10
        self.num_v = self.num_sub_bins * self.numBins
        self.v_base_res = self.v_range / self.num_v

        # === Compute voxelized canonical measurement === #
        pad_length = self.num_v
        self.pad_length = pad_length
        canon_voxel = np.zeros((self.num_y, self.num_x, self.num_v + 2*pad_length), dtype=np.float32)

        canon_x = np.linspace(x_min, x_max, self.num_x)
        canon_y = np.linspace(y_min, y_max, self.num_y)

        ones_vector = np.ones_like(canon_y)
        num_points = points.shape[0]

        # === Define pulse width parameters === #
        pulse_width_sigma = 1 # std in # of bins after fitting gaussian to pulse
        num_bins_pulse = int(np.ceil(pulse_width_sigma * 6 * self.num_sub_bins))
        x = np.linspace(-3*pulse_width_sigma, 3*pulse_width_sigma, num_bins_pulse)
        pulse_shape = np.exp(-x**2 / (2 * pulse_width_sigma**2)) * 10

        normal_vector = np.zeros((num_points, 3))
        normal_vector[:, 2] = 1
        normal_vector = normal_vector[:, None, :] # (num_points, 1, 3)

        if loaded_voxel is not None:
            canon_voxel = loaded_voxel
            print(f"Loaded canonical voxel from file")
        else:
            for i in tqdm(range(self.num_y), desc='Voxelizing Canonical Measurement'):
                with torch.no_grad():
                    # === Extract v index for pair of 2D spatial position on wall and each parabola === #
                    query_voxel = np.stack([canon_x, ones_vector * canon_y[i]], axis=-1) # (num_voxels, 2)
                    v_location = np.sum((query_voxel[None, :, :] - points[:, None, 0:2])**2, axis=-1) # (num_parabolas, num_voxels)
                    v_location += points[:, 2].reshape(-1, 1) ** 2
                    v_idx = np.floor(v_location / self.v_base_res).astype(int) # (num_parabolas, num_voxels)
                    v_idx = v_idx + self.pad_length # (num_parabolas, num_voxels)

                    # === Compute Lambertian falloff === #
                    # # get locations on wall
                    # wall_locs = np.zeros((self.num_x, 3)) # (num_voxels, 3)
                    # wall_locs[:, 0:2] = np.copy(query_voxel) 

                    # # get locations on object, assuming object is at z = 0.73
                    # points_3d = np.zeros((num_points, 3))
                    # points_3d[:, 0:2] = points[:, 0:2]
                    # points_3d[:, 2] = 0.73 # use r^2 falloff where obj is at z = 0.73
                    
                    # # compute lighting direction
                    # light_dir = points_3d[:, None, :] - wall_locs[None, :, :] # (num_points, num_voxels, 3)
                    # light_dir = light_dir / np.linalg.norm(light_dir, axis=-1, keepdims=True)

                    # cos_falloff = np.sum(light_dir * normal_vector, axis=-1) # (num_points, num_voxels)
                    # cos_falloff = np.clip(cos_falloff, 0, 1)
                    # cos_falloff = cos_falloff.reshape(-1, ) # (num_parabolas * num_voxels, )

                    # === Get x and y indices === #
                    x_idx = np.arange(self.num_x).reshape(1, -1) # (1, num_voxels)
                    x_idx = np.tile(x_idx, (num_points, 1)) # (num_parabolas, num_voxels)

                    y_idx = np.ones(self.num_x, dtype=int).reshape(1, -1) * i # (1, num_voxels)
                    y_idx = np.tile(y_idx, (num_points, 1)) # (num_parabolas, num_voxels)

                    # === Flatten indices === # 
                    x_idx = x_idx.reshape(-1) # (num_parabolas * num_voxels, )
                    y_idx = y_idx.reshape(-1) # (num_parabolas * num_voxels, )
                    v_idx = v_idx.reshape(-1) # (num_parabolas * num_voxels, )

                    # === Compute weights based on distance to parabola === #
                    # weights = 1 / np.clip(v_location.reshape(-1), 0.5, 25) # (num_paslacrabolas * num_voxels, )
                    weights = np.ones_like(x_idx)
                    # weights = cos_falloff
                    # print(np.min(weights), np.max(weights))

                    # if np.isnan(weights).any():
                    #     raise ValueError(f'NaN weights found in voxelization, {weights}')
                    # else:
                    #     print(np.min(weights), np.max(weights))

                    # === Add to canonical === #
                    # for k in range(num_bins_pulse):
                    #     # for l in range(5):
                    #     np.add.at(canon_voxel, (y_idx, x_idx, v_idx + k - num_bins_pulse//2), pulse_shape[k] * weights)
                    
                    for k in range(self.num_sub_bins):
                        # for l in range(5):
                        np.add.at(canon_voxel, (y_idx, x_idx, v_idx + k - self.num_sub_bins//2), weights)

        # === Normalize canonical measurement === #
        # canon_voxel = canon_voxel / np.sum(canon_voxel, axis=-1, keepdims=True)
        # canon_voxel = (canon_voxel > 0).astype(np.float32) # (num_y, num_x, num_v)
        self.canon_voxel = torch.Tensor(canon_voxel).to(device, non_blocking=True)

    def forward(self, pt_cloud: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        """
        Renders measurement at frame t from canonical measurement.

        Parameters:
        -----------
        pt_cloud   : virtual pixel locations on the wall in world coordinates 
                        (num_pixels, 2)
        deltas    : per-frame (x, y, v) shifts (batch_size, 3)

        Returns:
        --------
        hists  : rendered measurement (batch_size, num_y, num_x, numBins)

        """
        # start_time = time.time()
        num_pixels = pt_cloud.shape[0]
        batch_size = deltas.shape[0]
        
        # === Determine points on wall to sample === #
        x_samp = pt_cloud[:, 0].unsqueeze(0).repeat(batch_size, 1) 
        x_samp -= deltas[:, 0].unsqueeze(1).repeat(1, num_pixels) # (batch_size, num_pixels)
        y_samp = pt_cloud[:, 1].unsqueeze(0).repeat(batch_size, 1) 
        y_samp -= deltas[:, 1].unsqueeze(1).repeat(1, num_pixels) # (batch_size, num_pixels)

        if self.device == 'mps':
            x_samp_cpu = x_samp.detach().to('cpu', non_blocking=True)
            y_samp_cpu = y_samp.detach().to('cpu', non_blocking=True)

            if torch.max(x_samp_cpu) < self.x_min or torch.min(x_samp_cpu) > self.x_max:
                raise ValueError(f'x_samp out of bounds: {torch.min(x_samp_cpu)} {torch.max(x_samp_cpu)}')
            if torch.max(y_samp_cpu) < self.y_min or torch.min(y_samp_cpu) > self.y_max:
                raise ValueError(f'y_samp out of bounds: {torch.min(y_samp_cpu)} {torch.max(y_samp_cpu)}')
        else:
            if torch.max(x_samp) < self.x_min or torch.min(x_samp) > self.x_max:
                raise ValueError(f'x_samp out of bounds: {torch.min(x_samp)} {torch.max(x_samp)}')
            if torch.max(y_samp) < self.y_min or torch.min(y_samp) > self.y_max:
                raise ValueError(f'y_samp out of bounds: {torch.min(y_samp)} {torch.max(y_samp)}')

        # end_time = time.time()
        # print(f'hists: {end_time - start_time:.2f} seconds')

        # === Convert sampled (x, y) points to index === #
        # convert continuous location to discrete index
        x_samp_idx = self.num_x * (x_samp - self.x_min) / (self.x_max - self.x_min) # (batch_size, num_pixels)
        y_samp_idx = self.num_y * (y_samp - self.y_min) / (self.y_max - self.y_min) # (batch_size, num_pixels)

        # extend tensor for every timing bin
        x_samp_idx = x_samp_idx.unsqueeze(-1).repeat(1, 1, self.numBins) # (batch_size, num_pixels, numBins)
        y_samp_idx = y_samp_idx.unsqueeze(-1).repeat(1, 1, self.numBins) # (batch_size, num_pixels, numBins)

        # convert to int and flatten tensor
        x_samp_idx = x_samp_idx.int().reshape(batch_size * num_pixels * self.numBins) # (batch_size * num_pixels * numBins)
        y_samp_idx = y_samp_idx.int().reshape(batch_size * num_pixels * self.numBins) # (batch_size * num_pixels * numBins)

        # === Convert v shift to index === #
        x = torch.linspace(0, self.v_range, self.numBins).to(self.device, non_blocking=True)
        v_samp = torch.zeros(1, num_pixels).to(self.device, non_blocking=True)

        v_samp = v_samp - deltas[:, 2].reshape(batch_size, 1) # (batch_size, num_pixels)
        v_samp = v_samp.unsqueeze(-1) + x.reshape(1, 1, -1) # (batch_size, num_pixels, num_v)
        
        v_samp_idx = self.pad_length + (v_samp / self.v_base_res).int() # (batch_size, num_pixels, numBins)     
        v_samp_idx = v_samp_idx.reshape(batch_size * num_pixels * self.numBins) # (batch_size * num_pixels * numBins)

        # === Extract histograms at sampled (x, y, v) locations === #
        hists = self.canon_voxel[y_samp_idx, 
                                 x_samp_idx, 
                                 v_samp_idx] # (batch_size * num_pixels, numBins)

        hists = hists.reshape(batch_size, num_pixels, self.numBins) # (batch_size, num_pixels, num_v)

        return hists # (batch_size, num_pixels, numBins)
    
