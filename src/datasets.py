from torch.utils.data import IterableDataset, get_worker_info


class TensorIterableDataset(IterableDataset):
    def __init__(self, data, start=0, end=None, transform=None):
        super().__init__()

        if transform:
            self._data = transform(data)
        else:
            self._data = data

        self.start = start
        if not end:
            end = len(data)
        self.end = end

    def __len__(self):
        return self.end

    def __iter__(self):
        worker_info = get_worker_info()

        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
        else:
            # in a worker process
            # split workload
            per_worker = int(
                math.ceil((self.end - self.start) / float(worker_info.num_workers))
            )
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)

        return iter(self._data[iter_start:iter_end])
