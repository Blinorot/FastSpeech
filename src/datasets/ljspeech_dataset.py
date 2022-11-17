from src.utils import get_data_to_buffer
from torch.utils.data import Dataset


class LJDataset(Dataset):
    def __init__(self, data_path, mel_ground_truth, alignment_path,
                       text_cleaners, batch_expand_size, limit=None):
        self.buffer = get_data_to_buffer(data_path, mel_ground_truth, alignment_path,
                                         text_cleaners, batch_expand_size)
        if limit is not None:
            self.buffer = self.buffer[:limit]

        self.length_dataset = len(self.buffer)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        return self.buffer[idx]
