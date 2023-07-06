#
# Copyright (c) 2023 Tencent, inc, 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#

import io
import logging
import math
import os
import pprint
import sys
import time
import json
import pdb 
from tqdm import tqdm
from datetime import datetime

import transformers
import torch

from datasets.apps_dataset_codegen_online_v1 import APPSBaseDataset
from trainers.trainer_rl_online_v1 import Trainer_RL
from transformers import Trainer
from trainers.modeling_codegen import CodeGenForCausalLM
from trainers.modeling_t5 import T5ForConditionalGeneration

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def run_training(args, train_data):
    if args.model in ['codet5-base', 'codet5-large', 'codet5-large-ntp-py']:
        model_path = args.model_path if args.model_path is not None else 'models/{}'.format(args.model)  # Salesforce
        print("Loading model from {}...".format(model_path))
        model = T5ForConditionalGeneration.from_pretrained(
            model_path,
            tuning_mode=args.tuning_mode, 
            clone_rl_head=args.clone_rl_head) 
        
        if args.clone_rl_head:
            # Optional: clone a seperate RL head and initialize the model weights from finetuned LM head 
            print("Initializing RL head with finetuned LM head...")
            lm_head_params = model.lm_head.weight.detach().numpy()
            model.rl_head.weight = torch.nn.Parameter(torch.tensor(lm_head_params))

    elif args.model in ['codegen-2B-multi', 'codegen-2B-mono', 'codegen-350M-mono']:
        if args.model_path is None:
            model_path = 'models/{}'.format(args.model)
        else:
            model_path = args.model_path
        print("Loading codegen model from {}...".format(model_path))
        # model = AutoModelForCausalLM.from_pretrained(model_path)
        model = CodeGenForCausalLM.from_pretrained(model_path, tuning_mode=args.tuning_mode)
                
    print('Finished loading model {}'.format(args.model))

    start_iteration = 0
    train_data.start_iteration = start_iteration
    print(f"Starting main loop")

    training_args = transformers.TrainingArguments(
        output_dir=args.save_dir,
        overwrite_output_dir=True, 
        
        do_train=True,
        do_eval=False,
        do_predict=True,
        evaluation_strategy='no',
        eval_steps=0, 

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size_per_replica,
        gradient_accumulation_steps=args.grad_acc_steps,

        learning_rate=args.lr,
        weight_decay=0.05,
        lr_scheduler_type='polynomial',  #constant_with_warmup
        warmup_steps=500,

        logging_dir=args.save_dir, 
        logging_first_step=True,
        logging_steps=args.log_freq,
        save_steps=args.save_freq,
        save_total_limit=args.save_total_limit,

        dataloader_drop_last=True,
        dataloader_num_workers=0 if args.db else 8,

        local_rank=args.local_rank,

        deepspeed=args.deepspeed,
        fp16=args.fp16,
    )
    
    if args.tuning_mode in ['critic', 'rl']:
        trainer = Trainer_RL(
            model=model,
            args=training_args,
            train_dataset=train_data,
            tuning_mode=args.tuning_mode,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
        )
    
    trainer.train(update_freq=args.update_freq, update_root=args.update_root)
    
    if args.local_rank == 0:
        model.save_pretrained(os.path.join(args.save_dir, "final_checkpoint"))


def get_dataset(args): 
    
    fnames = os.listdir(args.train_path)
    if 'data_log.txt' in fnames:
        fnames.remove('data_log.txt')
    
    # train in debugging mode with small data split 
    if args.db:
        fnames = fnames[:50]

    if args.model in ['codegen-2B-multi', 'codegen-2B-mono', 'codegen-350M-mono']:
        max_tokens = 1312
        max_src_tokens = 800  # 600
    else:
        max_tokens = 1024
        max_src_tokens = -1

    train_data = APPSBaseDataset(
        dataroot=args.train_path,
        problem_dirs=fnames,
        model=args.model,
        max_tokens=max_tokens,
        max_src_tokens=max_src_tokens,
        sample_mode=args.sample_mode,
        tuning_mode=args.tuning_mode,
        relative_returns=args.relative_returns,
        memory_length=args.memory_length,
        fine_grained_feedback=args.fine_grained_feedback,
        fine_grained_weight=args.fine_grained_weight,
        adaptive_feedback=args.adaptive_feedback,
        update_root=args.update_root,
        fine_grained_type=args.fine_grained_type,

    )

    return train_data


def main(args):
    os.environ["WANDB_MODE"] = "offline"
    args.save_dir = args.save_dir + args.save_dir_suffix

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load dataset 
    train_data = get_dataset(args)

    # Save args to file
    json.dump(argsdict, open(os.path.join(args.save_dir, "args.json"), 'w'))

    # Load and train model; save model checkpoints 
    run_training(args, train_data)


if __name__ == "__main__":
    from configs.train_configs import *
    
    main(args)
