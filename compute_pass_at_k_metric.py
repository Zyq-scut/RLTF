#
# Copyright (c) 2023, Tencent, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#

import argparse
import os

from evaluate.evaluator import CodeEval


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute pass@k metric')
    parser.add_argument('--results_path', type=str, default='outputs/test_results_fix/codegen_checkpoint-29000_0100_none_bs1x4_lr2e-05-100-wo-postprocess',
                        help='path for run_unit_tests results, where is the root of *.pkl file')
    parser.add_argument('--num_problem', type=int, default=5000, help='number of test problem')
    parser.add_argument('--k', type=str, default='1, 5, 10, 100', help='k in pass@$k$')

    args = parser.parse_args()
    args.k = [int(ki) for ki in args.k.split(',')]

    pkl_files = os.listdir(args.results_path)
    if len(pkl_files) == 0:
        print('empty results path!')

    for pkl_file in pkl_files:
        if '.pkl' not in pkl_file:
            print('file error: %s' % (pkl_file))

    # do evaluate
    print('------------------------------------------------------------------------------------')
    print('Evaluating!   Testing path: %s ' % (args.results_path))
    print('------------------------------------------------------------------------------------')
    evaluator = CodeEval()
    all_pass_at_k = evaluator.compute(
        results_path=args.results_path,
        num_problem=args.num_problem,
        k=args.k
    )
    for key, value in all_pass_at_k.items():
        print('pass@%d : %f' % (key, value))

    print('end')
