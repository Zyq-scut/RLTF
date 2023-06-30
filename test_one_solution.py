#
# '''
# Copyright (c) 2023 Tencent, inc, 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
# '''#
import json
import numpy as np
import os
import os.path 
import pprint
import glob 
from tqdm import tqdm
import pdb
import traceback 
import pickle as pkl 
from typing import List

import subprocess
import tempfile as tfile
import copy


from utils.testing_util import run_test


def eval_one_problems_online(problem_id, gen_codes, test_path, go_on=True, ratio_reward=False):
    real_index = problem_id
    problem = test_path + '{}'.format(str(real_index).zfill(4))
    print('Testing sample {}'.format(problem))

    programs = []
    for o_idx, o in tqdm(enumerate(gen_codes), total=len(gen_codes), ncols=0, leave=False):
        try:
            curr_results, curr_error, _, curr_sol = run_test(prob_path=problem, test=o, debug=False,
                                                             example_tests=False, go_on=go_on)
            fixed = []
            for e in curr_results:
                if isinstance(e, np.ndarray):
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                fixed.append(e)
            curr_results = fixed
            assert isinstance(curr_results, list)

            # print(curr_results)
            pass_ratio = -1
            assert len(curr_results) != 0
            if len(curr_results) == 0 or -2 in curr_results:
                final_result = -2
            elif -1 in curr_results:
                final_result = -1
            elif False in curr_results:
                final_result = False
                if ratio_reward:
                    pass_num = 0
                    for r in curr_results:
                        if r == True:
                            pass_num += 1
                    pass_ratio = pass_num / len(curr_results)
                else:
                    pass_ratio = 0
            else:
                final_result = True
            programs.append(
                {'code': o, 'result': final_result, 'pass_ratio': pass_ratio, 'error': curr_error[-1], 'sol': curr_sol})

        except Exception as e:
            print(f"test framework exception = {repr(e)}{e}\n")

    return programs

def eval_and_save_problems(args):
    problems = sorted(glob.glob(args.test_path + '/*'))
    test_indices = []
    for problem_idx, problem in enumerate(problems):
        problem_id = int(problem.split('/')[-1])
        code_file_path = args.code_path + '/{}.json'.format(problem_id)
        if os.path.exists(code_file_path):
            test_indices.append(problem_idx)

    real_index = test_indices[args.index]
    problem = problems[real_index]
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    print('Testing sample {}'.format(problem))
    
    if args.example_tests:
        print("Using example tests") 
    
    codes_loc = args.code_path + '/{}.json'.format(real_index)
    if not os.path.isfile(codes_loc):
        exit() 
    with open(codes_loc, "r") as file: 
        gen_codes = json.load(file)[str(real_index)]['code']

    test_file = os.path.join(problem, "input_output.json")
    tests = json.load(open(test_file, 'r'))
    nb_tests = len(tests['inputs'])
    if args.max_tests!=-1 and nb_tests > args.max_tests: 
        exit() 

    if os.path.isfile(args.output_path + '/{}.pkl'.format(real_index)):
        with open(args.output_path + '/{}.pkl'.format(real_index), 'rb') as f:
            results = pkl.load(f)[real_index]['results']
            if len(results) == len(gen_codes):
                exit()
            else:
                gen_codes.remove(gen_codes[len(results)])
                gen_codes.insert(len(results), '\npass\n')
        
    print("Saving to {}".format(args.output_path + '/{}.pkl'.format(real_index)))

    all_results, all_errors, all_sols = [], [], []

    for o_idx, gen_code in tqdm(enumerate(gen_codes), total=len(gen_codes), ncols=0, leave=False):

        curr_results = []
        curr_errors = []
        curr_sol = None
        try:

            curr_results, curr_errors, _, curr_sol = run_test(prob_path=problem, test=gen_code, debug=args.debug,
                                          example_tests=args.example_tests)

            # curr_errors = [(e, traceback.format_tb(e.__traceback__)) if e is not None else e for e in curr_errors]
            # fixed = []
            # for e in curr_results:
            #     if isinstance(e, np.ndarray):
            #         e = e.item(0)
            #     if isinstance(e, np.bool_):
            #         e = bool(e)
            #     fixed.append(e)
            # curr_results = fixed

        except Exception as e:
            print(f"test framework exception = {repr(e)}{e}\n")
            break
        finally:
            assert isinstance(curr_results, list)
            all_results.append(curr_results)
            all_errors.append(curr_errors)
            all_sols.append(curr_sol)

        save_results = {real_index : {'results': all_results, 'errors': all_errors, 'sols': all_sols}}
        with open(args.output_path + '/{}.pkl'.format(real_index), "wb") as file:
            pkl.dump(save_results, file)

    '''
    How to read results:
    [-2] = compile error, 
    [-1] = runtime error 
    [False] = failed test case 
    [True] = passed test case
    '''

    save_results = {real_index : {'results': all_results, 'errors': all_errors, 'sols': all_sols}} 
    pkl.dump(save_results,  open(args.output_path + '/{}.pkl'.format(real_index), "wb"))                    

def main(args):    
    argsdict = vars(args)    
    eval_and_save_problems(args)

if __name__ == "__main__":
    from configs.unit_test_configs import * 
    main(args)
