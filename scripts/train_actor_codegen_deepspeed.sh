## '''
# Copyright (c) 2023 Tencent, inc, 2022, salesforce.com, inc.
## All rights reserved.
## SPDX-License-Identifier: BSD-3-Clause
## For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
## '''##

USE_TF=NO deepspeed --master_port 62000 \
    train_codegen.py \
    --batch-size-per-replica 1 --grad-acc-steps 4 \
    --epochs 10 --lr 2e-5 \
    --save-freq 5000 --log-freq 10 --save_total_limit 5 \
    --fp16 \
    --tuning_mode none --model codegen-2B-mono \
    --deepspeed configs/deepspeed_configs.json 
    
