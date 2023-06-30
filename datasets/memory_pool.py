#
# Copyright (c) 2023, Tencent, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#

import numpy as np
import random


class Sample:
    def __init__(self):
        pass


class MemoryPool:
    def __init__(self, sample_list, max_length=-1):
        self.file_list = []
        self.memory_queue = [sample_list]
        self.max_length = -1
        self._update_sample_pool()
        self.max_length = max_length

    def update(self, new_samples):
        assert isinstance(new_samples, list)
        num_new_samples = len(new_samples)

        if num_new_samples > len(self.sample_pool):
            self.memory_queue = [new_samples]

        else:
            while num_new_samples > 0:
                old_samples = self.memory_queue[0]
                if len(old_samples) > num_new_samples:
                    random.shuffle(old_samples)
                    self.memory_queue[0] = old_samples[num_new_samples:]
                    num_new_samples = 0

                else:
                    self.memory_queue = self.memory_queue[1:]
                    num_new_samples -= len(old_samples)

            self.memory_queue.append(new_samples)
        self._update_sample_pool()

    def _update_sample_pool(self):
        self.sample_pool = []
        for sample_list in self.memory_queue:
            self.sample_pool = self.sample_pool + sample_list

        if self.max_length != -1:
            if len(self.sample_pool) > self.max_length:
                self.sample_pool = self.sample_pool[-self.max_length:]

        random.shuffle(self.sample_pool)
