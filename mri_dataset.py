import os
import torch
from pathlib import Path
from torch.utils.data import Dataset

class PreprocessedVolumeDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = Path(os.path.expanduser(data_dir))
        self.pt_files = sorted(self.data_dir.glob('*.pt'))
        assert len(self.pt_files) > 0, f"No .pt files found in {self.data_dir}"
        print(f"Found {len(self.pt_files)} volumes in {self.data_dir}")

    def __len__(self):
        return len(self.pt_files)

    def __getitem__(self, idx):
        data = torch.load(self.pt_files[idx], weights_only=False)
        return data['volume']
