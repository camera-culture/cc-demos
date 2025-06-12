import numpy as np
import torch
from torch import nn

from cc_hardware.utils.constants import TORCH_DEVICE


class DeepLocation8(nn.Module):
    """
    DeepLocation8 model: 2-layer convolutional network designed for 8x8 histogram input
    """

    def __init__(self, height: int, width: int, num_bins: int, out_dims: int = 2):
        super().__init__()

        self.height = height
        self.width = width
        self.num_bins = num_bins

        # in: (n, self.height, self.width, 16)
        self.conv_channels = 4
        self.conv_channels2 = 8
        self.conv3d = nn.Conv3d(
            in_channels=1,
            out_channels=self.conv_channels,
            kernel_size=(3, 3, 7),
            padding=(1, 1, 3),
        )
        # (n, 4, self.height, self.width, 16)
        self.batchnorm3d = nn.BatchNorm3d(self.conv_channels)
        self.batchnorm3d2 = nn.BatchNorm3d(self.conv_channels2)
        # reshape to (n, 4, self.height x self.width, 16)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        # (n, 4, self.height, self.width, 8)
        self.conv3d2 = nn.Conv3d(
            in_channels=self.conv_channels,
            out_channels=self.conv_channels2,
            kernel_size=(3, 3, 5),
            padding=(1, 1, 2),
        )
        # (n, 8, self.height, self.width, 8)
        # reshape to (n, 8, self.height x self.width, 8)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        # (n, 8, self.height, self.width, 4)

        feat_bins = num_bins // 4
        in_feats  = self.conv_channels2 * height * width * feat_bins
        self.fc1  = nn.Linear(in_feats, 128)

        self.fc1_bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, out_dims)  # 2 output dimensions (x, y)
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(p=0.7)

    def forward(self, x) -> torch.Tensor:
        x = self.relu(self.conv3d(x.unsqueeze(1)))
        x = self.batchnorm3d(x)
        x = torch.reshape(
            x,
            (x.shape[0], self.conv_channels * self.height * self.width, self.num_bins),
        )
        x = self.pool1(x)
        x = torch.reshape(
            x, (x.shape[0], self.conv_channels, self.height, self.width, -1)
        )
        x = self.relu(self.conv3d2(x))
        x = self.batchnorm3d2(x)
        x = torch.reshape(
            x, (x.shape[0], self.conv_channels2 * self.height * self.width, -1)
        )
        x = self.pool2(x)
        x = torch.reshape(
            x, (x.shape[0], self.conv_channels2, self.height, self.width, -1)
        )

        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc1_bn(x)
        x = self.fc2(x)
        return x

    def evaluate(self, x: torch.Tensor | np.ndarray | list) -> np.ndarray:
        """
        Evaluate the model on input data x.
        Args:
            x (torch.Tensor): Input tensor of shape (n, height, width, num_bins).
        Returns:
            torch.Tensor: Output tensor of shape (n, out_dims).
        """
        if isinstance(x, (np.ndarray, list)):
            x = torch.tensor(x, dtype=torch.float32, device=TORCH_DEVICE)
        elif not isinstance(x, torch.Tensor):
            raise TypeError(
                "Input must be a numpy array or a torch tensor."
                f" Got {type(x)} instead."
            )

        if x.ndim == 3:
            x.unsqueeze_(0)

        self.eval()
        with torch.no_grad():
            return self.forward(x).cpu().numpy()


class ModelWrapper(nn.Module):
    """Base wrapper that delegates `forward` to the wrapped model, then
    calls `process_output` for optional post-processing."""

    def __init__(self, model: nn.Module, queue=None):
        super().__init__()
        self.model = model.to(TORCH_DEVICE)
        self.queue = queue

    def forward(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        # --- unchanged conversion/validation ---
        if isinstance(x, (np.ndarray, list)):
            x = torch.tensor(x, dtype=torch.float32, device=TORCH_DEVICE)
        elif not isinstance(x, torch.Tensor):
            raise TypeError(f"Unsupported input type {type(x)}")

        # --- add missing batch-dim logic (matches DeepLocation8.evaluate) ---
        if x.ndim == 3:          # (H, W, B)
            x = x.unsqueeze(0)   # -> (1, H, W, B)

        # ---------------------------------------------------------------
        out = self.model(x)
        out = self.process_output(out)
        if self.queue is not None:
            self.queue.put(out.detach().cpu().numpy())
        return out

    def process_output(self, output: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Override in subclasses."""
        return output


class KalmanFilter:
    def __init__(self, state_dim: int, process_noise_var: float, measurement_noise_var: float):
        self.n = state_dim
        self.x = torch.zeros(self.n, 1, device=TORCH_DEVICE)
        self.P = torch.eye(self.n, device=TORCH_DEVICE)
        self.F = torch.eye(self.n, device=TORCH_DEVICE)
        self.Q = torch.eye(self.n, device=TORCH_DEVICE) * process_noise_var
        self.H = torch.eye(self.n, device=TORCH_DEVICE)
        self.R = torch.eye(self.n, device=TORCH_DEVICE) * measurement_noise_var

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: torch.Tensor):
        z = z.reshape(self.n, 1)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ torch.linalg.inv(S)
        y = z - self.H @ self.x
        self.x = self.x + K @ y
        self.P = (torch.eye(self.n, device=TORCH_DEVICE) - K @ self.H) @ self.P

    def get_state(self) -> torch.Tensor:
        return self.x.clone()


class KalmanWrapper(ModelWrapper):
    """Smooth model outputs using a simple Kalman Filter."""

    def __init__(
        self,
        model: nn.Module,
        queue=None,
        process_noise_var: float = 0.01,
        measurement_noise_var: float = 0.01,
    ):
        super().__init__(model, queue)
        self.kf = KalmanFilter(
            state_dim=2,
            process_noise_var=process_noise_var,
            measurement_noise_var=measurement_noise_var,
        )

    def process_output(self, output: torch.Tensor) -> torch.Tensor:  # noqa: D401
        # Assume batch size 1; use first row if batch > 1.
        meas = output[0] if output.ndim == 2 else output
        self.kf.predict()
        self.kf.update(meas.detach())
        return self.kf.get_state().flatten()

    def evaluate(self, x):
        """Keeps the DeepLocation8 .evaluate(…) contract."""
        self.eval()
        with torch.no_grad():
            out = self.forward(x)        # already smoothed
        return out.cpu().numpy()