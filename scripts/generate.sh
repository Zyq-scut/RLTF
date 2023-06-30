##
## Copyright (c) 2023 Tencent, inc, 2022, salesforce.com, inc.
## All rights reserved.
## SPDX-License-Identifier: BSD-3-Clause
## For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
##
model_path=exps/codet5-large-ntp_none_bs4x1_lr2e-05/final_checkpoint
tokenizer_path=models/codet5-base
test_path=data/APPS/test/

start=0
end=5000
num_seqs_per_iter=20
num_seqs=20
temp=0.6

output_path=outputs/codet5-large-ntp_none_bs4x1_lr2e-05_0020/

CUDA_VISIBLE_DEVICES=0 python generate.py \
    --model_path $model_path \
    --tokenizer_path $tokenizer_path \
    --test_path $test_path \
    --output_path $output_path \
    -s $start -e $end \
    --num_seqs $num_seqs --num_seqs_per_iter $num_seqs_per_iter \
    --temperature $temp