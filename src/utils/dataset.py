import torch
import torch.utils.data


class SlidingWindowDataset(torch.utils.data.Dataset):
    def __init__(self, *tensors, window=1, horizon=1, dtype=torch.float):
        super().__init__()
        assert all(tensors[0].shape[0] == t.shape[0] for t in tensors)

        self._tensors = tensors
        self._window = window
        self._horizon = horizon
        self._dtype = dtype
        self._transform = transform

    def __getitem__(self, index):
        item = []
        for t in self._tensors:
            x = t[index : index + self._window]
            y = t[index + self._window : index + self._window + self._horizon]
            item.append(
                {
                    "x": torch.from_numpy(x).type(self._dtype),
                    "y": torch.from_numpy(y).type(self._dtype),
                }
            )
        return item

    def __len__(self):
        return self._tensors[0].shape[0] - self._window - self._horizon + 1
