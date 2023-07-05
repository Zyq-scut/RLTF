#
# Copyright (c) 2023, Tencent, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#

import os
import argparse
import subprocess
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate program parallely.')
    parser.add_argument('--model_path', type=str, default='exps/codet5-large_rl_bs4x8_lr2e-06_offline_v1_goon_sourcelen800_fine-grained0/checkpoint-5000',
                        help='path for actor model')
    parser.add_argument('--tokenizer_path', type=str, default='./models/codet5-base', help='path for tokenizer')
    parser.add_argument('--test_path', type=str, default='./data/APPS/test/', help='path for test data')

    parser.add_argument('--start', type=int, default=0, help='start index of test samples')
    parser.add_argument('--end', type=int, default=5000, help='end index of test samples')
    parser.add_argument('--num_seqs_per_iter', type=int, default=20,
                        help='number of possible minibatch to generate programs per iteration, depending on GPU memory')
    parser.add_argument('--num_seqs', type=int, default=20, help='number of total generated programs per test sample')
    parser.add_argument('--temperature', type=float, default=0.6, help='temperature for sampling tokens')

    parser.add_argument('--output_path', type=str, default='./outputs',
                        help='path to save output programs')

    args = parser.parse_args()

    K = 8
    for k in range(K):
        model_name = args.model_path.split('/')[-1]
        output_path = os.path.join(args.output_path, model_name + f'_{args.num_seqs:04d}_rl_bs4x8_lr2e-06_offline_v1_goon_sourcelen800_fine-grained0-{k}')
        # output_path = args.output_path

        gpu_num = torch.cuda.device_count()
        interval = (args.end - args.start) // gpu_num
        print(f'the {k} th round. checkpoint: [{args.model_path}]')
        print(f'saving to [{output_path}]')
        print('available gpu number: %d  process interval: %d' % (gpu_num, interval))

        p = [0 for i in range(gpu_num)]
        for i in range(gpu_num):
            if i == gpu_num - 1:
                start = args.start + interval * i
                end = args.end
            else:
                start = args.start + interval * i
                end = args.start + interval * (i + 1)

            render_cmd = 'CUDA_VISIBLE_DEVICES=%d python generate.py ' \
                         '--model_path %s ' \
                         '--tokenizer_path %s ' \
                         '--test_path %s ' \
                         '--output_path %s ' \
                         '-s %d ' \
                         '-e %d ' \
                         '--num_seqs %d ' \
                         '--num_seqs_per_iter %d ' \
                         '--temperature %f' % (i, args.model_path, args.tokenizer_path, args.test_path, output_path,
                                               start, end, args.num_seqs, args.num_seqs_per_iter, args.temperature)
            p[i] = subprocess.Popen(render_cmd, shell=True)

        for pi in p:
            pi.wait()

    print('end')
