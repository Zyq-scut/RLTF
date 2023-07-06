#
# Copyright (c) 2023, Tencent, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#

import time
import os
import argparse
import torch
from generate_codegen import generate_critic_inputs, generate_prompt
from test_one_solution import eval_one_problems_online
import json
import pickle as pkl
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from trainers.modeling_codegen import CodeGenForCausalLM
import random
from copy import deepcopy
import signal
from program_feedback import ErrorFinder
import traceback

parser = argparse.ArgumentParser(description="")

parser.add_argument("--interval", default=600, type=int, help='max time one use model generate')
parser.add_argument("--model_path", type=str, default='exps/codegen-2B-mono_rl_bs1x32_lr2e-06_online_v1/', help='Path of trained model')
parser.add_argument("--base_path", type=str, default='exps/codegen_copy/checkpoint-29000', help='Path of based model')
parser.add_argument('--critic_path', default='models/codegen_finetuned_critic', type=str, help='path to cirtic model')
parser.add_argument("--train_path", default="data/APPS/train/", type=str, help='')
parser.add_argument("--output_path", default="online_data_codegen/", type=str, help='')
parser.add_argument("--tokenizer_path", default='models/codegen-2B-mono', type=str, help='Path to the tokenizer')
parser.add_argument('--test_example_path', type=str, default='data/APPS_test_example_tests/', help='path for test example data')
parser.add_argument("--num_seqs", default=10, type=int, help='Number of total generated programs per test sample')
parser.add_argument("--max_len", default=512, type=int, help='Maximum length of output sequence')
parser.add_argument('--source_len', default=800, type=int, help='Maximum length of input sequence')
parser.add_argument("--binary_prediction", default=0, type=int, help='if model is a critic model, enable this for binary classification i.e. passed test or failed test only')
parser.add_argument("--temperature", default=0.6, type=float, help='temperature for sampling tokens')
parser.add_argument('--random_temperature', type=bool, default=False, const=True, nargs='?', help='random temperature or not')
parser.add_argument("--adaptive_feedback", default=1, type=int, help='[-0.3, 1] adaptive feedback or not')
parser.add_argument("--fine_grained_feedback", default=1, type=int, help='fine-grained feedback or not')
parser.add_argument("--go_on", default=1, type=int, help='go on ot not')

args = parser.parse_args()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if not os.path.exists(args.output_path):
    os.mkdir(args.output_path)

candidate_temperature = [0.6, 0.7, 0.8, 0.9, 1.0]
errorfinder = ErrorFinder()


def generate_one_problem(model, tokenizer, critic_model, problem_id, device):
    problem_str = str(problem_id).zfill(4)
    prob_path = os.path.join(args.train_path, problem_str)
    test_case_path = os.path.join(prob_path, "input_output.json")
    if not os.path.exists(test_case_path):
        return

    prompt_path = os.path.join(prob_path, "question.txt")
    starter_path = os.path.join(prob_path, "starter_code.py")

    if not os.path.exists(starter_path):
        starter_path = None

    input_text, q_token_ids_1, q_token_ids_2 = generate_prompt(args, test_case_path, prompt_path,
                                                               None,
                                                               tokenizer, starter_path)
    with torch.no_grad():
        q_token_ids = q_token_ids_1 + q_token_ids_2
        if len(q_token_ids) > args.source_len:
            q_token_ids = q_token_ids_1[:args.source_len - len(q_token_ids_2)] + q_token_ids_2

        input_ids = torch.LongTensor(q_token_ids).unsqueeze(0).cuda()

        if args.random_temperature:
            gen_temperature = random.choice(candidate_temperature)
        else:
            gen_temperature = args.temperature

        gen_output_programs = []
        gen_output_ids = model.generate(
            input_ids,
            do_sample=True,
            temperature=gen_temperature,
            max_length=args.max_len + args.source_len,
            num_return_sequences=args.num_seqs,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        baseline_output_programs = []
        baseline_output_ids = model.generate(
            input_ids,
            do_sample=False,
            temperature=gen_temperature,
            max_length=args.max_len + args.source_len,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        for output_id in gen_output_ids:
            output_id = output_id[len(q_token_ids):]
            gen_output_programs.append(tokenizer.decode(output_id, skip_special_tokens=True))
        for output_id in baseline_output_ids:
            output_id = output_id[len(q_token_ids):]
            baseline_output_programs.append(tokenizer.decode(output_id, skip_special_tokens=True))


        baseline_results = eval_one_problems_online(problem_id, baseline_output_programs, args.train_path, go_on=bool(args.go_on), adaptive_feedback=bool(args.adaptive_feedback))
        gen_results = eval_one_problems_online(problem_id, gen_output_programs, args.train_path, go_on=bool(args.go_on), adaptive_feedback=bool(args.adaptive_feedback))

        input_texts, input_codes, gt_error_types = generate_critic_inputs(args, test_case_path, prompt_path, "", tokenizer, starter_path, solutions=gen_results)
        text_tensor = torch.tensor(input_texts).to(device)
        code_tensor = torch.tensor(input_codes).to(device)
        gt_error_tensor = torch.tensor(gt_error_types).to(device)
        curr_inputs = {'input_ids': text_tensor, 'error_types': gt_error_tensor, 'labels': code_tensor}
        _, error_preds, error_hidden_states = critic_model(**curr_inputs, return_error_hidden_states=True)
        assert len(gt_error_types) == len(error_preds)
        final_pass_ratio = []
        error_line, error_info, reward_type = [], [], []
        for r in gen_results:
            final_pass_ratio.append(r['pass_ratio'])
            if args.fine_grained_feedback:
                cur_error_line, cur_error_info, cur_reward_type = errorfinder.find_error_line(r['result'], r['error'], r['sol'])
                error_line.append(cur_error_line)
                error_info.append(cur_error_info)
                reward_type.append(cur_reward_type)

        saved_critic_scores = {'code': input_codes, 'prompt': input_texts,
                               'gt_error_type': gt_error_types,
                               'pred_error_type': error_preds.cpu().numpy(),
                               'error_hidden_states': error_hidden_states.cpu().numpy(),
                               'pass_ratio': final_pass_ratio,
                               'error_line': error_line,
                               'reward_type': reward_type,
                               'error_info': error_info
                               }

        baseline_codes_loc = os.path.join(args.output_path, problem_str, "baseline_solutions_final.json")
        if not os.path.exists(os.path.dirname(baseline_codes_loc)):
            os.mkdir(os.path.dirname(baseline_codes_loc))
        if os.path.exists(baseline_codes_loc):
            data = json.load(open(baseline_codes_loc, "r"))
            data.append({'code': baseline_results[0]['code'], 'result': baseline_results[0]['result'],
                         'pass_ratio': baseline_results[0]['pass_ratio']})
            json.dump(data, open(baseline_codes_loc, "w"))
        else:
            with open(baseline_codes_loc, "w") as f:
                print("baseline_codes_loc:", baseline_codes_loc)
                json.dump([{'code': baseline_results[0]['code'], 'result': baseline_results[0]['result'],
                            'pass_ratio': baseline_results[0]['pass_ratio']}], f)
        ii = 0
        while True:
            gen_codes_loc = os.path.join(args.output_path, problem_str, "gen_solutions_critic_scores_{}.pkl".format(str(ii)))
            if os.path.exists(gen_codes_loc):
                ii += 1
            else:
                print(gen_codes_loc)
                pkl.dump(saved_critic_scores, open(gen_codes_loc, 'wb'))
                break

def generate_all():
    no_io_programs = [1692, 1693, 1694, 1695, 1696, 1697, 1698, 1699, 1700, 1701, 1702, 1703, 1704, 1705, 1706, 1707,
                      1708, 1709, 1710, 1711, 1712, 1713, 1714, 1715, 1716, 1717, 1718, 1719, 1720, 1721, 1722, 1723,
                      1724, 1725, 1726, 1727, 1728, 1729, 1730, 1731, 1732, 1733, 1734, 1735, 1736, 1737, 1738, 1739,
                      1740, 1741, 1742, 1743, 1744, 1745, 1746, 1747, 1748, 1749, 1750, 1751, 1752, 1753, 1754, 1755,
                      1756, 1757, 1758, 1759, 1760, 1761, 1762, 1763, 1764, 1765, 1766, 1767, 1768, 1769, 1770, 1771,
                      1772, 1773, 1774, 1775, 1776, 1777, 1778, 1779, 1780, 1781, 1782, 1783, 1784, 1785, 1786, 1787,
                      1788, 1789, 1790, 1791, 1792, 1793, 1794, 1795, 1796, 1797, 1798, 1799, 1800, 1801, 1802, 1803,
                      1804, 1805, 1806, 1807, 1808, 1809, 1810, 1811, 1812, 1813, 1814, 1815, 1816, 1817, 1818, 1819,
                      1820, 1821, 1822, 1823, 1824, 1825, 1826, 1827, 1828, 1829, 1830, 1831, 1832, 1833, 1834, 1835,
                      1836, 1837, 1838, 1839, 1840, 1841, 1842, 1843, 1844, 1845, 1846, 1847, 1848, 1849, 1850, 1851,
                      1852, 1853, 1854, 1855, 1856, 1857, 1858, 1859, 1860, 1861, 1862, 1863, 1864, 1865, 1866, 1867,
                      1868, 1869, 1870, 1871, 1872, 1873, 1874, 1875, 1876, 1877, 1878, 1879, 1880, 1881, 1882, 1883,
                      1884, 1885, 1886, 1887, 1888, 1889, 1890, 1891, 1892, 1893, 1894, 1895, 1896, 1897, 1898, 1899,
                      1900, 1901, 1902, 1903, 1904, 1905, 1906, 1907, 1908, 1909, 1910, 1911, 1912, 1913, 1914, 1915,
                      1916, 1917, 1918, 1919, 1920, 1921, 1922, 1923, 1924, 1925, 1926, 1927, 1928, 1929, 1930, 1931,
                      1932, 1933, 1934, 1935, 1936, 1937, 1938, 1939, 1940, 1941, 1942, 1943, 1944, 1945, 1946, 1947,
                      1948, 1949, 1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 1963,
                      1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979,
                      1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995,
                      1996, 1997, 1998, 1999, 2360, 4758, 4759, 4760, 4761, 4762, 4763, 4764, 4765, 4766, 4767, 4768,
                      4769, 4770, 4771, 4772, 4773, 4774, 4775, 4776, 4777, 4778, 4779, 4780, 4781, 4782, 4783, 4784,
                      4785, 4786, 4787, 4788, 4789, 4790, 4791, 4792, 4793, 4794, 4795, 4796, 4797, 4798, 4799, 4800,
                      4801, 4802, 4803, 4804, 4805, 4806, 4807, 4808, 4809, 4810, 4811, 4812, 4813, 4814, 4815, 4816,
                      4817, 4818, 4819, 4820, 4821, 4822, 4823, 4824, 4825, 4826, 4827, 4828, 4829, 4830, 4831, 4832,
                      4833, 4834, 4835, 4836, 4837, 4838, 4839, 4840, 4841, 4842, 4843, 4844, 4845, 4846, 4847, 4848,
                      4849, 4850, 4851, 4852, 4853, 4854, 4855, 4856, 4857, 4858, 4859, 4860, 4861, 4862, 4863, 4864,
                      4865, 4866, 4867, 4868, 4869, 4870, 4871, 4872, 4873, 4874, 4875, 4876, 4877, 4878, 4879, 4880,
                      4881, 4882, 4883, 4884, 4885, 4886, 4887, 4888, 4889, 4890, 4891, 4892, 4893, 4894, 4895, 4896,
                      4897, 4898, 4899, 4900, 4901, 4902, 4903, 4904, 4905, 4906, 4907, 4908, 4909, 4910, 4911, 4912,
                      4913, 4914, 4915, 4916, 4917, 4918, 4919, 4920, 4921, 4922, 4923, 4924, 4925, 4926, 4927, 4928,
                      4929, 4930, 4931, 4932, 4933, 4934, 4935, 4936, 4937, 4938, 4939, 4940, 4941, 4942, 4943, 4944,
                      4945, 4946, 4947, 4948, 4949, 4950, 4951, 4952, 4953, 4954, 4955, 4956, 4957, 4958, 4959, 4960,
                      4961, 4962, 4963, 4964, 4965, 4966, 4967, 4968, 4969, 4970, 4971, 4972, 4973, 4974, 4975, 4976,
                      4977, 4978, 4979, 4980, 4981, 4982, 4983, 4984, 4985, 4986, 4987, 4988, 4989, 4990, 4991, 4992,
                      4993, 4994, 4995, 4996, 4997, 4999]

    can_be_chosed_programs_init = []
    for i in range(5000):
        if i not in no_io_programs:
            can_be_chosed_programs_init.append(i)
    can_be_chosed_programs = deepcopy(can_be_chosed_programs_init)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    critic_model = CodeGenForCausalLM.from_pretrained(args.critic_path, tuning_mode='critic')
    critic_model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
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

    while True:
        start = time.time()
        models = []
        if os.path.exists(args.model_path):
            models_raw = os.listdir(args.model_path)
            for m in models_raw:
                if 'checkpoint' in m:
                    models.append(os.path.join(args.model_path, m))
        if len(models) == 0:
            latest_model = args.base_path
            print(f'Using base model: {latest_model}')
            latest_model = AutoModelForCausalLM.from_pretrained(latest_model)
        else:
            models.sort(key=os.path.getctime, reverse=True)
            print(models)
            for path in models:
                try:
                    latest_model = AutoModelForCausalLM.from_pretrained(path)
                    print(f'Using latest model: {path}')
                    break
                except:
                    continue

        latest_model.to(device)
        print("load model time:", time.time() - start)

        while time.time() - start < args.interval:
            problem_id = random.sample(can_be_chosed_programs, 1)[0]
            try:
                can_be_chosed_programs.remove(problem_id)
                print(len(can_be_chosed_programs))
            except:
                print(problem_id, "can not delete")
            if len(can_be_chosed_programs) == 0:
                can_be_chosed_programs = deepcopy(can_be_chosed_programs_init)
            print("problem_id:", problem_id)

            try:
                time1 = time.time()
                generate_one_problem(latest_model, tokenizer, critic_model, problem_id, device)
                print("generate one problem time:", time.time() - time1)
            except Exception as e:
                print(traceback.format_exc())
                continue
        print("Time to change model!", time.time() - start)


generate_all()



