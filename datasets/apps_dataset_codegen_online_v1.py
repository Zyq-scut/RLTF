## '''
# Copyright (c) 2023 Tencent, inc, 2022, salesforce.com, inc.
## All rights reserved.
## SPDX-License-Identifier: BSD-3-Clause
## For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
## '''##

import torch
import glob
import logging
import random
import fnmatch
import numpy as np
import gc
import os
from tqdm import tqdm 
from collections import Counter
import pickle as pkl 
import json, pdb 

from multiprocessing import Manager
import transformers
from transformers import AutoTokenizer

import datasets.utils as dsutils
from datasets.memory_pool import MemoryPool
from program_feedback import find_position


class APPSBaseDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, problem_dirs, model, max_tokens, sample_mode, 
                 tuning_mode, max_src_tokens, relative_returns, memory_length=-1, fine_grained_feedback=False, fine_grained_weight=0, adaptive_feedback=False, update_root='', fine_grained_type=0):
        self.dataroot = dataroot
        self.problem_dirs = problem_dirs 

        self.model = model
        self.sample_mode = sample_mode
        self.tuning_mode = tuning_mode
        self.relative_returns = relative_returns
        
        self.max_tokens = max_tokens
        self.max_src_tokens = max_src_tokens
        self.memory_length = memory_length
        self.fine_grained_feedback = fine_grained_feedback
        self.fine_grained_weight = fine_grained_weight
        self.adaptive_feedback = adaptive_feedback
        self.online_root = update_root
        self.fine_grained_type = fine_grained_type

        self.samples = []  # sample (question, starter_code, solution, answer_type)
        self.all_error_types, self.all_error_subtypes, self.all_baseline_error_types = [], [], []
        self.initialize()

        if self.model in ['codegen-2B-multi', 'codegen-2B-mono']:
            tokenizer_path = 'models/{}'.format(self.model)
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.tokenizer.add_special_tokens(
                {
                    "bos_token": '<s>',
                    "eos_token": '</s>',
                    "unk_token": '<unk>',
                    "pad_token": '<pad>',
                    # "eod" : '<|endoftext|>', # no such token, but
                    "mask_token": '<mask>',
                }
            )
            self.eos_id = self.tokenizer.eos_token_id

        self.update_count = 0

    def load_gen_samples(self, sols, answer_type, starter_code, question_str):
        samples = []
        info = []
        
        for idx, sol in enumerate(sols):
            sol_str = sol['code']  
            sample = (question_str, starter_code, sol_str, answer_type, sol['result'], sol['error_type'])
            samples.append(sample)

            result = sol['result']
            error_type = sol['error_type']

            info.append((result, error_type))
            
        return samples, info
    
    def load_rl_samples(self, sols, baseline_error_type): 
        samples = []
        info = []
        
        for idx, code in enumerate(sols['code']):   
            samples.append((sols['prompt'][idx], code,
                            sols['gt_error_type'][idx], sols['error_hidden_states'][idx], baseline_error_type, None, None))
            info.append((sols['gt_error_type'][idx], sols['error_hidden_states'][idx], baseline_error_type))
            
        return samples, info

    def load_rl_samples_ratio(self, sols, baseline_error_type, baseline_pass_ratio):
        samples = []
        info = []

        for idx, code in enumerate(sols['code']):
            if not self.adaptive_feedback:
                baseline_pass_ratio = None
                sols['pass_ratio'][idx] = None
            if not self.fine_grained_feedback:
                samples.append((sols['prompt'][idx], code,
                                sols['gt_error_type'][idx], sols['error_hidden_states'][idx], baseline_error_type, sols['pass_ratio'][idx], baseline_pass_ratio))
            else:
                samples.append((sols['prompt'][idx], code, sols['gt_error_type'][idx], sols['error_hidden_states'][idx],
                                baseline_error_type, sols['pass_ratio'][idx], baseline_pass_ratio,
                                sols['error_line'][idx], sols['reward_type'][idx]
                                ))
            info.append((sols['gt_error_type'][idx], sols['error_hidden_states'][idx], baseline_error_type))

        return samples, info
     
    def load_gt_samples(self, sols, answer_type, starter_code, question_str):
        samples = []
        
        for sol_str in sols:
            sol_str = dsutils.reindent_code(sol_str)
            sample = (question_str, starter_code, sol_str, answer_type, 1, None)
            samples.append(sample)
        
        return samples 
    
    def get_gt_info(self):
        return (1, None)

    def get_baseline_error_type(self, sols): 
        return dsutils.get_error_type(sols[-1]['result'])

    # def get_baseline_error_type_ratio(self, sols):
    #     return dsutils.get_error_type(sols[-1]['result']), dsutils.get_error_type(sols[-1]['pass_ratio'])

    def update_error_stat(self, info):
        for i in info:
            error_type = dsutils.get_error_type(i[0])
            error_subtype = i[1]
            self.all_error_types.append(error_type)
            self.all_error_subtypes.append(error_subtype) 
            
    def update_error_stat_rl(self, info):
        for i in info:
            error_type = i[0]
            baseline_error_type = i[-1]
            self.all_error_types.append(error_type)
            self.all_baseline_error_types.append(baseline_error_type) 
    
    def initialize(self):
        all_samples = []
        skipped_problems = []
        samples_info = [] 
        gen_samples = []
        
        all_samples_dict = {}
        online_root = self.online_root
        if os.path.exists(self.online_root + '_old'):
            online_root = self.online_root + '_old'
        print(f"Loading {len(self.problem_dirs)} problems from {online_root}.")
        for problem_name in tqdm(self.problem_dirs):
            if self.tuning_mode in ['critic']:                
                gen_sols_fname = [os.path.join(self.dataroot, problem_name, "gen_solutions_2.json")]

            elif self.tuning_mode in ['rl']:
                gen_sols_fname = [os.path.join(online_root, problem_name, "gen_solutions_critic_scores_0.pkl")]
                
                if self.relative_returns: 
                    baseline_fname = os.path.join(online_root, problem_name, "baseline_solutions_final.json")
                
            question_fname = os.path.join(self.dataroot, problem_name, "question.txt")
            sols_fname = os.path.join(self.dataroot, problem_name, "solutions.json")
            input_output_fname = os.path.join(self.dataroot, problem_name, "input_output.json")

            if (not os.path.isfile(question_fname)) or (not os.path.isfile(sols_fname)):
                skipped_problems.append(problem_name)
                continue



            # Read the question description
            with open(question_fname, 'r') as f:
                question_str = f.read()
            
            starter_code = os.path.join(self.dataroot, problem_name, "starter_code.py")    
            if (os.path.isfile(starter_code)):
                answer_type = "\nUse Call-Based format\n"
                with open(starter_code, 'r') as f:
                    starter_code = f.read()
            else:
                answer_type = "\nUse Standard Input format\n"
                starter_code = ""

            sols_str_list = json.load(open(sols_fname, 'r'))
            gt_samples = self.load_gt_samples(sols_str_list, answer_type, starter_code, question_str)
            all_samples += gt_samples 
                
            # Read all the solutions
            if self.tuning_mode in ['critic']: 
                for fname in gen_sols_fname:
                    if os.path.exists(fname):
                        gen_sols = json.load(open(fname, 'r'))
                        samples, info = self.load_gen_samples(gen_sols, answer_type, starter_code, question_str) 
                        self.update_error_stat(info)
                        gen_samples += samples
                        samples_info += info
                
                # also include ground-truth samples to train critic model; assume success test outcomes 
                gen_samples += gt_samples
                info = [self.get_gt_info() for s in gt_samples]
                samples_info += info
                
            elif self.tuning_mode in ['rl']:
                if not os.path.isfile(gen_sols_fname[0]) or not os.path.isfile(input_output_fname):
                    skipped_problems.append(problem_name)
                    continue
                
                if self.relative_returns:
                    baseline_sample = json.load(open(baseline_fname, 'r'))
                    baseline_error_type = self.get_baseline_error_type(baseline_sample)
                    try:
                        baseline_pass_ratio = baseline_sample[-1]['pass_ratio']
                    except:
                        baseline_pass_ratio = None
                else:
                    baseline_error_type = -1 

                for fname in gen_sols_fname: 
                    if os.path.exists(fname):
                        gen_sols = pkl.load(open(fname, 'rb'))
                        #samples, info = self.load_rl_samples(gen_sols, baseline_error_type) # prompt, code, error, q, baseline_error, None, None
                        samples, info = self.load_rl_samples_ratio(gen_sols, baseline_error_type, baseline_pass_ratio)  # prompt, code, error, q, baseline_error, None, None
                        self.update_error_stat_rl(info)
                        gen_samples += samples
                        samples_info += info
                
        print(f"Loaded {len(all_samples)} samples from {self.dataroot}.")
        print(f"Skipped {len(skipped_problems)} problems from {self.dataroot}.")
        
        if self.tuning_mode in ['critic', 'rl']:
            if len(gen_samples) != len(samples_info): pdb.set_trace()
            print(f"Loaded {len(gen_samples)} generated samples from {self.dataroot}.")
            print("Error type distribution: {}".format(sorted(Counter(self.all_error_types).items())))
            if self.tuning_mode == 'critic':
                print("Error subtype distribution: {}".format(sorted(Counter(self.all_error_subtypes).items())))
            else:
                print("Baseline error distribution: {}".format(sorted(Counter(self.all_baseline_error_types).items())))
        else:
            print(f"Loaded {len(all_samples)} samples from {self.dataroot}.")
            
        self.samples = all_samples
        self.samples_info = samples_info
        self.memory_pool = MemoryPool(gen_samples, max_length=self.memory_length)
        self.gen_samples = self.memory_pool.sample_pool
        
        if self.relative_returns: 
            self.all_error_types = np.array(self.all_error_types)
            self.all_baseline_error_types = np.array(self.all_baseline_error_types)
            print('Sampled Error > Baseline error: {}/{}'.format((self.all_error_types>self.all_baseline_error_types).sum(),
                                                                  len(self.all_error_types)))
            print('Sampled Error = Baseline error: {}/{}'.format((self.all_error_types==self.all_baseline_error_types).sum(),
                                                                  len(self.all_error_types)))
            print('Sampled Error < Baseline error: {}/{}'.format((self.all_error_types<self.all_baseline_error_types).sum(),
                                                                  len(self.all_error_types)))
            
            sample_rewards = np.array([dsutils.get_reward_from_error_type(e) for e in self.all_error_types])
            baseline_rewards = np.array([dsutils.get_reward_from_error_type(e) for e in self.all_baseline_error_types])
            print("Relative returns distribution: {}".format(sorted(Counter(sample_rewards-baseline_rewards).items())))

    def update_sample(self, dataroot):
        gen_samples = []
        gen_sols_fname = []
        self.update_count += 1

        for problem_name in self.problem_dirs:
            problem_root = os.path.join(dataroot, problem_name)
            if not os.path.exists(problem_root):
                continue
            file_list = [os.path.join(problem_root, file) for file in os.listdir(problem_root)]
            for file in file_list:
                if file in self.memory_pool.file_list or 'gen_solutions_critic_scores' not in file:
                    continue
                else:
                    gen_sols_fname.append(file)
                    self.memory_pool.file_list.append(file)

        for fname in gen_sols_fname:
            #problem_name = fname.split('/')[:-1]
            problem_name = os.path.dirname(fname)
            baseline_fname = os.path.join(problem_name, 'baseline_solutions_final.json')
            if not os.path.exists(baseline_fname):
                continue
            baseline_sample = json.load(open(baseline_fname, 'r'))
            #baseline_error_type = self.get_baseline_error_type(baseline_sample)
            baseline_error_type = self.get_baseline_error_type(baseline_sample)
            baseline_pass_ratio = baseline_sample[-1]['pass_ratio']

            if os.path.exists(fname):
                gen_sols = pkl.load(open(fname, 'rb'))
                #samples, _ = self.load_rl_samples(gen_sols, baseline_error_type)  # prompt, code, error, q, baseline_error, None, None
                samples, _ = self.load_rl_samples_ratio(gen_sols, baseline_error_type, baseline_pass_ratio)  # prompt, code, error, q, baseline_error, gen_pass_ratio, baseline_pass_ratio

                gen_samples += samples
                #print('fname {}: {}'.format(fname, samples))

        self.memory_pool.update(gen_samples)
        self.gen_samples = self.memory_pool.sample_pool

        t = dsutils.get_reward_from_error_type

        gen_mean_reward = [t(sample[2]) for sample in gen_samples]
        gen_mean_reward_w_baseline = [t(sample[2]) - t(sample[4]) for sample in gen_samples]
        gen_mean_reward = np.mean(gen_mean_reward) if gen_mean_reward is not [] else None
        gen_mean_reward_w_baseline = np.mean(gen_mean_reward_w_baseline) if gen_mean_reward_w_baseline is not [] else None

        mean_reward = [t(sample[2]) for sample in self.gen_samples]
        mean_reward_w_baseline = [t(sample[2]) - t(sample[4]) for sample in self.gen_samples]
        mean_reward = np.mean(mean_reward) if mean_reward is not [] else None
        mean_reward_w_baseline = np.mean(mean_reward_w_baseline) if mean_reward_w_baseline is not [] else None

        log_text = f'update times <{self.update_count}> | find <{len(gen_sols_fname)}> new files | update <{len(gen_samples)}> samples\n'
        log_text += f'\tall_mean_reward <{mean_reward}> | all_mean_reward_w_baseline <{mean_reward_w_baseline}>\n'
        log_text += f'\tgen_mean_reward <{gen_mean_reward}> | gen_mean_reward_w_baseline <{gen_mean_reward_w_baseline}>\n'

        with open(os.path.join(dataroot, 'data_log.txt'), 'a+') as f:
            f.write(log_text)

        print(log_text)
        return gen_mean_reward, gen_mean_reward_w_baseline, mean_reward, mean_reward_w_baseline, len(gen_sols_fname), len(gen_samples)

    def __len__(self):
        if self.tuning_mode in ['critic', 'rl']:
            return len(self.gen_samples)
        return len(self.samples)

    def __getitem__(self, idx):
        
        if self.tuning_mode in ['critic']:
            raw_samples = self.pack_samples(idx, 'gen')
            inputs = self.sample_task(raw_samples, 'gen')
         
        elif self.tuning_mode in ['rl']:
            gt_sample_idx = random.randint(0, len(self.samples)-1)
            raw_gt_samples = self.pack_samples(gt_sample_idx)
            inputs = self.sample_task(raw_gt_samples)
            
            item = self.gen_samples[idx]
            # print('fffffffffffffffffff:  ', type(item), len(item), type(item[0]), type(item[3]))
            # info = self.samples_info[idx]
            gen_inputs = self.sample_rl_task(item)
            for k,v in gen_inputs.items():
                inputs['rl_{}'.format(k)] = v 
            
        else:
            raw_samples = self.pack_samples(idx)
            inputs = self.sample_task(raw_samples)

        gc.collect()
        return inputs

    def pack_samples(self, idx, sample_type=None):
        """
        Repeatedly pick question, answer pairs from self.dataroot until we hit max_tokens.
        This will not include the tokens for the QUESTION and ANSWER prompt, as well as the  
        self.question_prefix. These will be added later and the total input will be 
        truncated if necessary.

        Always include the sample at idx at the beginning.
        """
        curr_num_tokens = 0
        curr_samples = [] 
        
        if sample_type == 'gen':
            sample_pool = self.gen_samples
        else:
            sample_pool = self.samples
        
        if self.sample_mode == 'uniform_sol':
            curr_q, curr_s, curr_a, curr_q_prefix, curr_result, curr_error_subtype = sample_pool[idx]
            # if self.tuning_mode in ['critic'] and sample_type=='gen':
            #     curr_result, curr_error_subtype = self.samples_info[idx]
                
        elif self.sample_mode == 'uniform_prob':
            raise NotImplementedError()

        while curr_num_tokens < self.max_tokens:

            curr_q = curr_q[:150000]
            curr_s = curr_s[:150000]
            curr_a = curr_a[:150000]

            curr_num_tokens += len(self.tokenizer.tokenize(curr_q))
            curr_num_tokens += len(self.tokenizer.tokenize(curr_s))            
            curr_num_tokens += len(self.tokenizer.tokenize(curr_a))
            
            if self.tuning_mode in ['critic'] and sample_type=='gen': 
                curr_samples.append((curr_q, curr_s, curr_a, curr_q_prefix, 
                                     curr_result, curr_error_subtype))
                break
     
            else:
                curr_samples.append((curr_q, curr_s, curr_a, curr_q_prefix))
                
                # only pack 1 sample each sequence for codeT5 
                if self.model in ['codet5-base', 'codet5-large', 'codet5-large-ntp-py']:
                    break
                elif self.model in ['codegen-2B-multi', 'codegen-2B-mono']:
                    break

            if self.sample_mode == 'uniform_sol':
                new_idx = random.randint(0, len(sample_pool)-1)
                curr_q, curr_s, curr_a, curr_q_prefix = sample_pool[new_idx] 
            elif self.sample_mode == 'uniform_prob':
                raise NotImplementedError()

        return curr_samples

    def sample_task(self, samples, sample_type=None):

        input_ids = []
        label_ids = []
                    
        if self.tuning_mode in ['critic'] and sample_type == 'gen': 
            error_types = [] 
                    
        for sample in samples:
            if self.tuning_mode in ['critic'] and sample_type == 'gen': 
                q_str, s_str, a_str, answer_type, result, error_subtype = sample
            else:
                q_str, s_str, a_str, answer_type = sample

            q_str_part1 = "\nQUESTION:\n" + q_str
            q_str_part2 = "\n" + s_str + "\n" + answer_type + "\nANSWER:\n"
            q_token_ids_1 = self.tokenizer.encode(q_str_part1, verbose=False)
            q_token_ids_2 = self.tokenizer.encode(q_str_part2, verbose=False)
            answer_token_ids = self.tokenizer.encode(a_str, verbose=False)

            if len(q_token_ids_1) + len(q_token_ids_2) + len(answer_token_ids) > self.max_tokens:
                q_token_ids_1 = q_token_ids_1[:self.max_src_tokens - len(q_token_ids_2)]
            question_token_ids = q_token_ids_1 + q_token_ids_2

            answer_token_ids.append(self.tokenizer.eos_token_id)
            input_ids.extend(question_token_ids)
            label_ids.extend([-100] * len(question_token_ids))
            input_ids.extend(answer_token_ids)
            label_ids.extend(answer_token_ids)

            if self.tuning_mode in ['critic'] and sample_type == 'gen':
                error_types.append(dsutils.get_error_type(result))
                
        # Sanity checks and padding 
        input_ids_max_len = self.max_src_tokens \
            if self.model in ['codet5-base', 'codet5-large', 'codet5-large-ntp-py']\
            else self.max_tokens

        if len(input_ids) < input_ids_max_len:
            # input_ids
            if self.model in ['alphacode']:
                new_input_ids = [self.tokenizer.pad_token_id] * input_ids_max_len  # pad with pad_token_id
                new_input_ids[:len(input_ids)] = input_ids
                input_ids = new_input_ids
            elif self.model in ['codet5-base', 'codet5-large', 'codet5-large-ntp-py', 'codegen-2B-multi', 'codegen-2B-mono', 'alpaca', 'llama']:
                new_input_ids = [self.tokenizer.eos_token_id] * input_ids_max_len  # pad with eos_token_id
                new_input_ids[:len(input_ids)] = input_ids
                input_ids = new_input_ids

            # label_ids
            if self.model in ['codegen-2B-multi', 'codegen-2B-mono', 'alpaca', 'llama']:
                new_label_ids = [-100] * input_ids_max_len
                new_label_ids[:len(label_ids)] = label_ids
                label_ids = new_label_ids

        if self.model in ['codet5-base', 'codet5-large', 'codet5-large-ntp-py'] and \
                len(label_ids) < self.max_tokens:
            new_label_ids = [-100] * self.max_tokens
            new_label_ids[:len(label_ids)] = label_ids
            label_ids = new_label_ids

        if self.model in ['codegen-2B-multi', 'codegen-2B-mono', 'alpaca', 'llama'] and len(input_ids) != len(label_ids):
            # pdb.set_trace()
            raise ValueError("for decoder-only models, input_ids and label_ids should have the same length")

        # Cut off the excess
        input_ids = input_ids[:input_ids_max_len]
        label_ids = label_ids[:self.max_tokens]

        out_sample = {
            "input_ids": torch.LongTensor(input_ids),  # input + labels  vs  input
            "labels": torch.LongTensor(label_ids)  # -100 * len(input) + labels  vs  labels
        }

        return out_sample

    def sample_rl_task(self, item):
        # 重要!!!!!生成 sample 的时候，保存的 input_ids 只要 input; label 只要 label，且没有end of token，在这里再 pad;
        if not self.fine_grained_feedback:
            input_ids, label_ids, gt_error, error_logit, baseline_error, gen_pass_ratio, baseline_pass_ratio = item
        else:
            input_ids, label_ids, gt_error, error_logit, baseline_error, gen_pass_ratio, baseline_pass_ratio, error_line, reward_type = item

            fine_grained_error_mask = find_position(self.tokenizer, label_ids, error_line, reward_type).astype(float)

        # gt_error, error_logit, baseline_error = info

        softmax_fn = torch.nn.Softmax(dim=-1)
        rewards = softmax_fn(torch.tensor(error_logit))[:, gt_error]
        # rewards = rewards * 0 + 1

        if self.relative_returns:
            curr_reward = dsutils.get_reward_from_error_type(gt_error, gen_pass_ratio)
            if self.fine_grained_feedback and self.fine_grained_type == 1:
                t = torch.zeros_like(rewards)
                t[:len(fine_grained_error_mask)] = self.fine_grained_weight * len(fine_grained_error_mask) / len(
                    fine_grained_error_mask == 1) * torch.tensor(fine_grained_error_mask)
                curr_reward = curr_reward - t
                # print("curr_reward:", curr_reward)
                # print("t:", t)
                # print("curr_reward:", curr_reward.shape)
                # print("t:", t.shape)
            baseline_reward = dsutils.get_reward_from_error_type(baseline_error, baseline_pass_ratio) if baseline_error!=-1 else 0
            relative_reward = curr_reward - baseline_reward
            rewards *= relative_reward
        else:
            rewards *= dsutils.get_reward_from_error_type(gt_error)


        if self.fine_grained_feedback and self.fine_grained_type == 0:
            t = torch.zeros_like(rewards)
            t[:len(fine_grained_error_mask)] = self.fine_grained_weight * len(fine_grained_error_mask) / len(fine_grained_error_mask == 1) * torch.tensor(fine_grained_error_mask)
            rewards = rewards - t
            # print("rewards:", rewards)
            # print("t:", t)
            # print("rewards:", rewards.shape)
            # print("t:", t.shape)

        # masking rewards
        reward_mask = (error_logit == -np.float('Inf'))[:,0]
        rewards[reward_mask] = 0.0
        rl_label_ids = np.array(label_ids)
        rl_label_ids[reward_mask] = -100

        assert len(label_ids) == len(rewards)
        assert len(input_ids) == len(label_ids)

        if len(input_ids) < self.max_tokens:
            # pad input ids
            new_input_ids = [self.tokenizer.eos_token_id] * self.max_tokens
            new_input_ids[:len(input_ids)] = input_ids
            input_ids = new_input_ids

            # pad label ids and reward
            new_rl_label_ids = [-100] * self.max_tokens
            new_rl_label_ids[:len(label_ids)] = label_ids
            rl_label_ids = new_rl_label_ids

            new_rewards = torch.zeros(self.max_tokens)
            new_rewards[:len(rewards)] = rewards
            rewards = new_rewards

        # cut off
        input_ids = input_ids[:self.max_tokens]
        rewards = rewards[:self.max_tokens]
        rl_label_ids = rl_label_ids[:self.max_tokens]

        out_sample = {
            "input_ids": torch.LongTensor(input_ids),
            "rewards": rewards,
            'label_ids': torch.LongTensor(rl_label_ids)
        }

        return out_sample

    def debug(self):
        print('max length of input_ids: [%d]' % (self.max_src_tokens))

        n_too_long = 0
        for idx in range(len(self)):
            sample = self.pack_samples(idx)

            q_str, s_str, gt_code_str, answer_type, gen_code_str, result, error_subtype = sample[0]

            q_str = "\nQUESTION:\n" + q_str + "\n" + s_str + "\n" + answer_type + "\n" + gen_code_str + \
                    "\n" + str(result) + "\n" + error_subtype + "\nANSWER:\n"

            input_ids = self.tokenizer.encode(q_str, verbose=False)

            if len(input_ids) > self.max_src_tokens:
                n_too_long += 1

        print('number of all samples: [%d]' % (len(self)))
        print('number of the samples that are too long: [%d]' % (n_too_long))

        # idx = np.random.randint(0, len(self))
        # sample = self.pack_samples(idx)
        # q_str, s_str, gt_code_str, answer_type, gen_code_str, result, error_subtype = sample[0]
        # q_str = "\nQUESTION:\n" + q_str + "\n" + s_str + "\n" + answer_type + "\n" + gen_code_str + \
        #         "\n" + str(result) + "\n" + error_subtype + "\nANSWER:\n"
        #
        # print(idx)
        # print(q_str)

    
if __name__ == '__main__':
    max_tokens = 512
    max_src_tokens = 600
    model = 'codet5-large'

    train_path = 'data/APPS/train/'
    fnames = os.listdir(train_path)

    # train_data = APPSRepairDataset(
    #     dataroot=train_path,
    #     problem_dirs=fnames,
    #     model=model,
    #     max_tokens=max_tokens,
    #     max_src_tokens=max_src_tokens,
    # )

    train_data = APPSBaseDataset(
        dataroot=train_path,
        problem_dirs=fnames,
        model=model,
        max_tokens=max_tokens,
        max_src_tokens=max_src_tokens,
        sample_mode='uniform_sol',
        tuning_mode=None,
        relative_returns=False,
    )

    train_data.debug()
