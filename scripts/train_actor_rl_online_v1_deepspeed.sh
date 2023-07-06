## '''
# Copyright (c) 2023 Tencent, inc, 2022, salesforce.com, inc.
## All rights reserved.
## SPDX-License-Identifier: BSD-3-Clause
## For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
## '''##

USE_TF=NO deepspeed --master_port 62000 \
    train_online_v1.py \
    --batch-size-per-replica 4 --grad-acc-steps 8 \
    --epochs 200 --lr 2e-6 \
    --save-freq 50 --log-freq 10 --save_total_limit 1000 \
    --fp16 \
    --tuning_mode rl --model codet5-large \
    --model_path exps/codet5-large_none_bs4x2_lr2e-05_800_512/checkpoint-16000 \
    --relative_returns \
    --deepspeed configs/deepspeed_configs.json \
    --update_freq 50 \
    --update_root online_data \
    --memory_length 6400 \
    --save_dir_suffix _online_v1 \
    --fine_grained_feedback \
    --fine_grained_weight 0.3 \
    --fine_grained_type 1 \
    --adaptive_feedback