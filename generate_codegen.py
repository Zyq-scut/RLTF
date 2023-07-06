#
# Copyright (c) 2023 Tencent, inc, 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
# 
import json
import os
import pprint
import torch
import pdb 
import glob 
from tqdm import tqdm
import pickle as pkl 
import numpy as np 
from collections import Counter 
from transformers import RobertaTokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM
from trainers.modeling_codegen import CodeGenForCausalLM
import datasets.utils as dsutils


def generate_prompt(args, test_case_path, prompt_path, solutions_path, tokenizer, 
                    starter_path=None):
    
    _input = "\nQUESTION:\n"
    q_str_part1 = "\nQUESTION:\n"
    with open(prompt_path, "r") as f:  # question.txt
        data = f.readlines()
        data = "".join(data)
    _input += data
    q_str_part1 += data

    q_str_part2 = ''
    if starter_path != None:
        with open(starter_path, "r") as f:  # starter.py
            data = f.readlines()
            data = "".join(data)
            data = "\n" + data 
        _input += data
        q_str_part2 += data
    
    if os.path.exists(test_case_path):  # input_output.json
        with open(test_case_path, "r") as f:
            data = json.load(f)
        if not data.get("fn_name"):
            _input += "\nUse Standard Input format"
            q_str_part2 += "\nUse Standard Input format"
        else:
            _input += "\nUse Call-Based format"
            q_str_part2 += "\nUse Call-Based format"

    elif starter_path is not None and os.path.exists(starter_path):
        _input += "\nUse Call-Based format"
        q_str_part2 += "\nUse Call-Based format"
    else:
        _input += "\nUse Standard Input format"
        q_str_part2 += "\nUse Standard Input format"
        
    _input += "\nANSWER:\n"
    q_str_part2 += "\nANSWER:\n"

    q_token_ids_1 = tokenizer.encode(q_str_part1, verbose=False)
    q_token_ids_2 = tokenizer.encode(q_str_part2, verbose=False)
    
    return _input, q_token_ids_1, q_token_ids_2


def generate_critic_inputs(args, test_case_path, prompt_path, solutions_path, tokenizer,
                           starter_path=None, gt_solutions=False, solutions=None):
    _input, q_token_ids_1, q_token_ids_2 = generate_prompt(args, test_case_path, prompt_path, solutions_path, tokenizer,
                                                           starter_path)

    #solutions = json.load(open(solutions_path, 'r'))
    if solutions == None:
        solutions = json.load(open(solutions_path, 'r'))

    all_texts = []
    gt_errors = []
    all_codes = []

    for sol_index, solution in enumerate(solutions):
        input_ids = []
        label_ids = []

        if gt_solutions:
            solution_str = dsutils.reindent_code(solution)
        else:
            solution_str = dsutils.reindent_code(solution['code'])

        # codegen rl
        answer_token_ids = tokenizer.encode(solution_str)

        question_token_ids = q_token_ids_1 + q_token_ids_2
        if len(question_token_ids) > args.source_len:
            question_token_ids = q_token_ids_1[:args.source_len - len(q_token_ids_2)] + q_token_ids_2

        answer_token_ids.append(tokenizer.eos_token_id)
        input_ids.extend(question_token_ids)
        label_ids.extend([-100] * len(question_token_ids))
        input_ids.extend(answer_token_ids)
        label_ids.extend(answer_token_ids)

        assert len(input_ids) == len(label_ids)

        if len(input_ids) < (args.max_len + args.source_len):
            # input_ids
            new_input_ids = [tokenizer.eos_token_id] * (args.max_len + args.source_len)  # pad with eos_token_id
            new_input_ids[:len(input_ids)] = input_ids
            input_ids = new_input_ids

            # label_ids
            new_label_ids = [-100] * (args.max_len + args.source_len)
            new_label_ids[:len(label_ids)] = label_ids
            label_ids = new_label_ids

        all_texts.append(input_ids)
        all_codes.append(label_ids)

        if gt_solutions:
            gt_errors.append(dsutils.get_error_type(True, binary=args.binary_prediction))
        else:
            gt_errors.append(dsutils.get_error_type(solution['result'], binary=args.binary_prediction))

    return all_texts, all_codes, gt_errors


def main(args):

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    original_problems = glob.glob(args.test_path + '/*')
    problems = sorted(original_problems) 

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)
    print("Saving results to {}".format(args.output_path))

    if args.start > len(problems) or args.start < 0:
        print(f"start index {args.start} > number of problems {len(problems)}")
        return
    start = args.start
    if args.end is None or args.end > len(problems):
        end = len(problems)
    else:
        end = args.end
    problems = problems[start:end]
    
    # Set up model
    print("Loading model from {}...".format(args.model_path))
    if args.critic_scores:
        model = CodeGenForCausalLM.from_pretrained(args.model_path, tuning_mode='critic')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained('models/codegen-2B-mono')
    tokenizer.add_special_tokens(
        {
            "bos_token": '<s>',
            "eos_token": '</s>',
            "unk_token": '<unk>',
            "pad_token": '<pad>',
            # "eod" : '<|endoftext|>', # no such token, but
            "mask_token": '<mask>',
        }
    )
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
   
    if args.critic_scores:
        all_preds = [] 
        all_gts = [] 
        
    # main eval loop
    for index, problem in tqdm(enumerate(problems), ncols=0, total=len(problems)):
        
        prob_path = os.path.join(problem)
        print(f"problem path = {prob_path}")
        
        problem_id = int(problem.split('/')[-1])
        
        '''
        if args.critic_scores and \
            os.path.exists(os.path.join(args.output_path, f"{problem_id}_gt{args.gt_solutions}.pkl")):
            continue 
        elif os.path.exists(os.path.join(args.output_path, f"{problem_id}.json")):
            continue 
        '''
        
        test_case_path = os.path.join(prob_path, "input_output.json")
        prompt_path = os.path.join(prob_path, "question.txt")
        starter_path = os.path.join(prob_path, "starter_code.py")
        if args.critic_scores and not args.gt_solutions: 
            solutions_path = os.path.join(prob_path, "gen_solutions.json")
        else:
            solutions_path = os.path.join(prob_path, "solutions.json")
        if not os.path.exists(starter_path):
            starter_path = None

        if args.critic_scores:
            input_texts, input_codes, gt_error_types = \
                generate_critic_inputs(args, test_case_path, prompt_path, solutions_path,
                                       tokenizer, starter_path, args.gt_solutions)
        else:
            input_text, q_token_ids_1, q_token_ids_2 = generate_prompt(args, test_case_path, prompt_path, solutions_path,
                                          tokenizer, starter_path)

        with torch.no_grad():
            if args.critic_scores:
                text_tensor = torch.tensor(input_texts).to(device)
                code_tensor = torch.tensor(input_codes).to(device)
                gt_error_tensor = torch.tensor(gt_error_types).to(device)
                
                curr_inputs = {'input_ids': text_tensor, 'error_types': gt_error_tensor, 'labels': code_tensor}
                _, error_preds, error_hidden_states = model(**curr_inputs, return_error_hidden_states=True)
                # error_hidden_states 是和 input_ids 对应的
                
                assert len(gt_error_types) == len(error_preds)
                all_preds.extend(error_preds.cpu().numpy().tolist())
                all_gts.extend(gt_error_types)
                
                saved_critic_scores = {'code': input_codes, 'prompt': input_texts,
                                          'gt_error_type': gt_error_types, 
                                          'pred_error_type': error_preds.cpu().numpy(),
                                          'error_hidden_states': error_hidden_states.cpu().numpy()}
                
                if args.gt_solutions:
                    scores_loc = os.path.join(prob_path,  "solutions_critic_scores.pkl")
                else:
                    scores_loc = os.path.join(prob_path,  "gen_solutions_critic_scores.pkl")
                    
                pkl.dump(saved_critic_scores, open(scores_loc, 'wb'))

            else:
                q_token_ids = q_token_ids_1 + q_token_ids_2
                if len(q_token_ids) > args.source_len:
                    q_token_ids = q_token_ids_1[:args.source_len - len(q_token_ids_2)] + q_token_ids_2

                input_ids = torch.LongTensor(q_token_ids).unsqueeze(0).cuda()

                num_loops = int(args.num_seqs / args.num_seqs_per_iter)
                output_programs = []
                output_ids_list = []
                for i in tqdm(range(num_loops), ncols=0, total=num_loops, leave=False):
                    output_ids = model.generate(
                        input_ids, 
                        do_sample=True, 
                        temperature=args.temperature, 
                        max_length=args.max_len + args.source_len,
                        num_return_sequences=args.num_seqs_per_iter,
                        top_p=0.95,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )

                    for output_id in output_ids:
                        output_id = output_id[len(q_token_ids):]
                        output_programs.append(tokenizer.decode(output_id, skip_special_tokens=True))
                        # output_ids_list.append(output_id.tolist())

                saved_codes = {}
                saved_codes[problem_id] = {'code': output_programs, 'prompt': input_text, 'output_id': output_ids_list}

                codes_loc = os.path.join(args.output_path, f"{problem_id}.json")
                with open(codes_loc, "w") as f:
                    json.dump(saved_codes, f)

    if args.critic_scores: 
        print("Total number of samples: {}".format(len(all_gts)))
        acc = (np.array(all_preds) == np.array(all_gts)).sum()/len(all_gts)
        print("Error Pred Acc: {}".format(acc))
        print("Prediction distribution: {}".format(Counter(all_preds)))
        print("GT distribution: {}".format(Counter(all_gts)))
                    
if __name__ == "__main__":
    
    from configs.generate_configs import * 
    
    main(args)
