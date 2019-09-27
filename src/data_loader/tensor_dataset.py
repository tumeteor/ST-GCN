from torch.utils.data import Dataset
from functools import lru_cache


class CustomTensorDataset(Dataset):
    def __init__(self, tensors, adj_list, mask_list, time_steps):
        self.tensors = tensors
        self.adj_list = adj_list
        self.mask_list = mask_list
        self.time_steps = time_steps
        self._index_mapping()

    @lru_cache(maxsize=8)
    def __getitem__(self, index):
        graph_idx = self.index_map[index]
        print(f"graph_idx :{graph_idx}, time_steps: {self.time_steps}")
        print(f"BBB: {self.mask_list[index].shape}")
        print(f"mask list len: {len(self.mask_list)}")
        print(f"tensor len: {len(self.tensors)}")
        return tuple((self.tensors[index][0], self.tensors[index][1], self.adj_list[graph_idx], self.mask_list[index]))

    @lru_cache(maxsize=8)
    def __len__(self):
        return self.time_steps * len(self.adj_list)

    @lru_cache(maxsize=8)
    def _index_mapping(self):
        self.index_map = dict()
        for idx in range(self.time_steps * len(self.adj_list)):
            self.index_map[idx] = int(idx / self.time_steps)

    def _load_tensor_from_path(self):
        # TODO
        pass
