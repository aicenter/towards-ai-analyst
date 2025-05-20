import random
from typing import Iterator

import numpy as np
from torch.utils.data import Sampler


class WeightedSampler(Sampler):
    def __init__(self, *, labels, positive_ratio: float = 0.1, max_num_samples: int | None) -> None:
        self.labels = labels
        self.num_samples = max_num_samples if max_num_samples else len(labels)
        self.positive_ratio = positive_ratio
        self.positive_indexes = np.arange(len(labels))[self.labels == 0].tolist()
        self.negative_indexes = np.arange(len(labels))[self.labels == 1].tolist()

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        """
        Custom iterator with stratified sampling.

        Yields:
            int: Sampled index
        """
        # Track samples to ensure we don't exceed num_samples
        samples_generated = 0

        while samples_generated < self.num_samples:
            # Determine sample type based on positive ratio
            r = random.random()

            if r < self.positive_ratio:
                # Sample from positive indexes
                if not self.positive_indexes:
                    continue
                ix = random.choice(self.positive_indexes)
            else:
                # Sample from negative indexes
                if not self.negative_indexes:
                    continue
                ix = random.choice(self.negative_indexes)

            yield ix
            samples_generated += 1
