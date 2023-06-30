## '''
# Copyright (c) 2023 Tencent, inc, 2022, salesforce.com, inc.
## All rights reserved.
## SPDX-License-Identifier: BSD-3-Clause
## For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
## '''##

USE_TF=NO deepspeed --master_port 62000 \
    train.py \
    --batch-size-per-replica 4 --grad-acc-steps 8 \
    --epochs 20 --lr 2e-6 \
    --save-freq 1000 --log-freq 10 --save_total_limit 20 \
    --fp16 \
    --tuning_mode rl --model codet5-large \
    --model_path exps/codet5-large_none_bs4x2_lr2e-05/checkpoint-16000 \
    --relative_returns \
    --deepspeed configs/deepspeed_configs.json
