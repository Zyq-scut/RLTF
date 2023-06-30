#
# Copyright (c) 2023, Tencent, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#

import numpy as np
import pickle
import os
from tqdm import tqdm


class CodeEval:
    def __init__(self):
        pass

    def _pass_at_k(self, n, c, k):
        """
        :param n: total number of samples
        :param c: number of correct samples
        :param k: k in pass@$k$
        """
        if n - c < k: return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    def compute(self, results_path, num_problem=5000, k=[1, 5, 10]):
        '''
        : param results_path: fold for results
        : param k: list(), each k in pass@$k$
        '''
        all_pass_at_k = dict()
        for ki in k:
            all_pass_at_k[ki] = []

        pkl_files = os.listdir(results_path)
        if len(pkl_files) != num_problem:
            print('file number [%d] is not equal to problem number [%d]' % (len(pkl_files), num_problem))
            print('padding pass_at_k metric with 0')

            for ki in k:
                all_pass_at_k[ki] = [0 for _ in range(num_problem - len(pkl_files))]

        for pkl_file in tqdm(pkl_files):
            with open(os.path.join(results_path, pkl_file), 'rb') as f:
                results = pickle.load(f)[int(pkl_file.replace('.pkl', ''))]['results']

            n = len(results)
            c = 0
            for result in results:
                if not result:
                    print(pkl_file)
                    continue

                if (np.array(result) == True).all():
                    c += 1
                else:
                    continue

            for ki in k:
                pass_at_k = self._pass_at_k(n, c, ki)
                all_pass_at_k[ki].append(pass_at_k)

        for ki in k:
            all_pass_at_k[ki] = np.mean(all_pass_at_k[ki])

        return all_pass_at_k
