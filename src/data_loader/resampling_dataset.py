from torch.utils.data import Sampler


class CustomSampler(Sampler):

    def __init__(self, data_source, cum_indices, shuffle=True):
        super().__init__(data_source)
        self.data_source = data_source
        self.cum_indices = [0] + cum_indices
        self.shuffle = shuffle

    def __iter__(self):
        batch = []
        for prev, curr in zip(self.cum_indices, self.cum_indices[1:]):
            for idx in range(prev, curr):
                batch.append(idx)
            yield batch
            batch = []

    def __len__(self):
        return len(self.data_source)

