import torch
from torch.utils.data import DataLoader, Dataset, random_split

from cc_hardware.drivers.spads import SPADDataType

class PoseCaptureDataset(Dataset):
    def __init__(self, captures, h, w, bins, window: int = 1):
        self.cap = captures
        self.h, self.w, self.bins = h, w, bins
        self.win = max(window, 1)

    def __len__(self):
        return len(self.cap)

    def _avg_hist(self, idx: int) -> torch.Tensor:
        half = self.win // 2
        lo = max(0, idx - half)
        hi = min(len(self.cap), idx + half + 1)
        h_arr = [
            torch.tensor(self.cap[i][SPADDataType.HISTOGRAM], dtype=torch.float32)
            for i in range(lo, hi)
        ]
        return torch.mean(torch.stack(h_arr), dim=0)

    def __getitem__(self, idx):
        f = self.cap[idx]
        pose_label = torch.tensor(f["pose"])
        hist = (
            self._avg_hist(idx)
            if self.win > 1
            else torch.tensor(f[SPADDataType.HISTOGRAM], dtype=torch.float32)
        )
        if hist.ndim == 1:
            hist = hist.view(self.h, self.w, self.bins)
        return pose_label, hist


def pose_collate(batch):
    poses, hists = zip(*batch)
    return torch.stack(poses), torch.stack(hists)


def create_pose_dataloaders(captures, split=0.8, batch_size=32, **kwargs):
    dataset = PoseCaptureDataset(captures, **kwargs)
    train_size = int(len(dataset) * split)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pose_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pose_collate)
    return train_loader, val_loader
