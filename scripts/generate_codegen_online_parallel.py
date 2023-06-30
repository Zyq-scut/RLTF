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
import glob
import time


if __name__ == '__main__':
    gpu_num = torch.cuda.device_count()
    print('available gpu number: %d  ' % (gpu_num))
    p = [0 for i in range(gpu_num)]
    while True:
        for i in range(gpu_num):
            if p[i] == 0:
                render_cmd = 'CUDA_VISIBLE_DEVICES=%d python generate_codegen_online.py ' % (i)
                p[i] = subprocess.Popen(render_cmd, shell=True)
            else:
                if p[i].poll() is not None:
                    print("generate_codegen_online.py exited unexpectedly, restarting...")
                    render_cmd = 'CUDA_VISIBLE_DEVICES=%d python generate_codegen_online.py ' % (i)
                    p[i] = subprocess.Popen(render_cmd, shell=True)
        time.sleep(60)
