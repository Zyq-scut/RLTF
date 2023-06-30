#
# Copyright (c) 2023, Tencent, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#

import numpy as np
import os
import pickle
import re


class ErrorFinder:
    def __init__(self):
        pass

    def find_error_line(self, res, err, sol):
        temp = err
        if not isinstance(err, str):
            err = str(err)

        if res == -2:
            if isinstance(temp, SyntaxError):
                error_line, error_info, reward_type = self.find_ce(err, sol)
            else:
                error_line = ''
                error_info = ''
                reward_type = 'ignore'
        elif res == -1:
            error_line, error_info, reward_type = self.find_re(err, sol)
        else:
            error_line, error_info, reward_type = self.find_false_and_true(err, sol)

        return error_line, error_info, reward_type

    def find_ce(self, err, sol):
        error_info = err.split(' (')[0]
        if sol[-1] == '\n':
            sol = sol[:-1]

        if 'inconsistent use of tabs and spaces in indentation' in err or \
                'unindent does not match any outer indentation' in err or \
                'EOF while scanning triple-quoted string literal' in err or \
                'duplicate argument' in err or 'import * only allowed at module level' in err:
            error_line = ''
            reward_type = 'ignore'

        elif 'cannot import name' in err:
            reward_type = 'line'

            pattern = r'cannot import name \'(.*)\' from'
            matches = re.findall(pattern, err)
            m = matches[0]

            sol_lines = sol.split('\n')
            for line in sol_lines:
                if m in line:
                    error_line = line.lstrip().rstrip()
                    break

        elif 'is parameter and nonlocal' in err:
            reward_type = 'line'

            pattern = r'name \'(.*)\' is parameter and nonlocal'
            matches = re.findall(pattern, err)
            m = matches[0]

            sol_lines = sol.split('\n')
            for line in sol_lines:
                if 'nonlocal' in line:
                    split_line = line.replace(',', ' ').lstrip().rstrip().split(' ')
                    if m in split_line:
                        error_line = line.lstrip().rstrip()
                        break

        elif 'no binding for nonlocal' in err:
            reward_type = 'line'

            pattern = r'no binding for nonlocal \'(.*)\' found'
            matches = re.findall(pattern, err)
            m = matches[0]

            sol_lines = sol.split('\n')
            for line in sol_lines:
                if 'nonlocal' in line:
                    split_line = line.replace(',', ' ').lstrip().rstrip().split(' ')
                    if m in split_line:
                        error_line = line.lstrip().rstrip()
                        break
        elif 'No module named' in err:
            reward_type = 'line'

            pattern = r'No module named \'(.*)\''
            matches = re.findall(pattern, err)
            m = matches[0]

            sol_lines = sol.split('\n')
            for line in sol_lines:
                if 'import' in line:
                    split_line = line.replace(',', ' ').lstrip().rstrip().split(' ')
                    if m in split_line:
                        error_line = line.lstrip().rstrip()
                        break

        else:
            pattern = r', line (\d+)'
            matches = re.findall(pattern, err)

            assert len(matches) == 1, f'err:{err}\nsol:\n{sol}'
            line_num = int(matches[0])

            sol_lines = sol.split('\n')
            error_line = sol_lines[line_num - 1]

            if 'unexpected EOF while parsing' in err or 'invalid syntax' in err or \
                    'EOL while scanning string literal' in err:
                if line_num == len(sol_lines):
                    error_line = ''
                    reward_type = 'whole'
                else:
                    error_line = error_line.lstrip().rstrip()
                    reward_type = 'line'

            else:
                reward_type = 'line'
                if 'unexpected indent' in err:
                    error_line = error_line
                else:
                    error_line = error_line.lstrip().rstrip()

        return error_line, error_info, reward_type

    def find_re(self, err, sol):
        if err[-1:] == '\n':
            err = err[:-1]

        err_list = err.split('\n')
        error_info = err_list[-1]

        if 'RuntimeError' in err or err == '' or 'error returncode' in err:
            error_line = ''
            reward_type = 'ignore'

        elif 'TimeoutException' in err or 'RecursionError' in err:
            error_line = ''
            reward_type = 'whole'

        else:
            assert 'line' in err, err
            error_line = ''
            reward_type = 'ignore'

            while len(err_list) > 1:
                error_line_temp = err_list.pop().lstrip().rstrip()
                if error_line_temp in sol:
                    error_line = error_line_temp
                    reward_type = 'line'
                    break

        return error_line, error_info, reward_type

    def find_false_and_true(self, err, sol):
        error_line = ''
        error_info = None
        reward_type = 'ignore'
        return error_line, error_info, reward_type


def find_position(tokenizer, text_tokens, tgt_text, reward_type, debug=False):

    if reward_type == 'ignore':
        tgt_mask = np.zeros(len(text_tokens)).astype(bool)
        if debug:
            return tgt_mask, ''
        else:
            return tgt_mask

    elif reward_type == 'whole':
        tgt_mask = np.ones(len(text_tokens)).astype(bool)

        if debug:
            decode_mask_text = ''
            for token, m in zip(text_tokens, tgt_mask):
                if m:
                    decode_mask_text += tokenizer.decode(token)

            return tgt_mask, decode_mask_text
        else:
            return tgt_mask

    elif reward_type == 'line':
        tgt_mask = np.zeros(len(text_tokens))

        for idx, token in enumerate(text_tokens):
            if token == -100:
                text_tokens = text_tokens[:idx]
                break

        # double pointer
        start_p = tokenizer.decode(text_tokens).find(tgt_text) + 1
        # assert start_p, f'decode text:\n{tokenizer.decode(text_tokens)}\ntgt_text:\n{tgt_text}'
        if not start_p:
            if debug:
                return tgt_mask, tgt_text
            else:
                return tgt_mask

        if tgt_text != '':
            for idx, t in enumerate(text_tokens):
                decode_text = tokenizer.decode(text_tokens[:idx + 1])

                if len(decode_text) < start_p:
                    continue
                else:
                    tgt_mask[idx] = 1
                    if tgt_text not in decode_text:
                        continue
                    else:
                        break

        tgt_mask = tgt_mask.astype(bool)

        if debug:
            decode_mask_token = []
            for idx, (token, m) in enumerate(zip(text_tokens, tgt_mask)):
                if m:
                    decode_mask_token.append(token)
                    # decode_mask_text = tokenizer.decode(text_tokens[:idx+1])
            decode_mask_text = tokenizer.decode(decode_mask_token)
            # print(f'Mask:{len(tgt_mask)}\n{tgt_mask}\nDecode Mask:\n{decode_mask_text}')
            return tgt_mask, decode_mask_text
        else:
            return tgt_mask
