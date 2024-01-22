import json
import os
import subprocess
import re
import threading

import numpy
from tqdm import tqdm
import numpy as np
from itertools import chain
import sys
sys.path.append('/home/wpd/back_fortran_subroutine')
import utils
import unit_test_utils


def pass_at_k(n, c, k):
    """
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@$k$
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
def get_compile_pass(code_list):
    correct_num = 0
    total_num = len(code_list)
    result_list = [unit_test_utils.check_iscompile(code) for code in tqdm(code_list)]
    for result in result_list:
        if result==True:
            correct_num+=1
    return correct_num/total_num
def extract_functions(fortran_code):
    fortran_code = fortran_code.lower()
    # 使用正则表达式匹配以"!"开头且包含"function"或"Function"的行
    pattern = re.compile(r'^\s*![^:\n]*\b(?:function|Function)\b.*$', re.MULTILINE)
    # 通过替换去除匹配到的行
    cleaned_code = re.sub(pattern, '', fortran_code)
    functions = []
    function_code = ''
    function_flag = 0
    for code_line in cleaned_code.split('\n'):
        if ('function' in code_line and 'test_function' not in code_line) and ('end function' not in code_line):
            function_flag+=1
            function_code+=code_line+'\n'
        if  ('end function' in code_line):
            function_flag -= 1
            function_code += code_line + '\n'
            if function_flag==0:
                functions.append(function_code)
                function_code = ''
        if ('function' not in code_line) and ('end function' not in code_line) and function_flag!=0:
            function_code+=code_line+'\n'
    # function_pattern = r'\bfunction\b.*?\bend function\b'
    # functions = re.findall(function_pattern, cleaned_code, re.DOTALL | re.IGNORECASE)
    if len(functions) > 0:
        return functions[0]
    else:
        return ''
def extract_subroutines(fortran_code):
    fortran_code = fortran_code.lower()
    # 使用正则表达式匹配以"!"开头且包含"function"或"Function"的行
    pattern = re.compile(r'^\s*![^:\n]*\b(?:subroutine|Subroutine)\b.*$', re.MULTILINE)
    # 通过替换去除匹配到的行
    cleaned_code = re.sub(pattern, '', fortran_code)
    functions = []
    function_code = ''
    function_flag = 0
    for code_line in cleaned_code.split('\n'):
        if ('subroutine' in code_line and 'test_subroutine' not in code_line) and ('end subroutine' not in code_line):
            function_flag+=1
            function_code+=code_line+'\n'
        if  ('end subroutine' in code_line):
            function_flag -= 1
            function_code += code_line + '\n'
            if function_flag==0:
                functions.append(function_code)
                function_code = ''
        if ('subroutine' not in code_line) and ('end subroutine' not in code_line) and function_flag!=0:
            function_code+=code_line+'\n'
    # function_pattern = r'\bfunction\b.*?\bend function\b'
    # functions = re.findall(function_pattern, cleaned_code, re.DOTALL | re.IGNORECASE)
    if len(functions)>0:
        return functions[0]
    else:
        return ''
def extract_fortran(fortran_code):
    pattern = r"```fortran(.*?)```"
    matches = re.findall(pattern, fortran_code, re.DOTALL)
    if matches:
        extracted_code = matches[0]
        return extracted_code
    else:
        return fortran_code
def run_code(unit_code_list,code_list):
    correct_num = 0
    total_num = 0
    for i in tqdm(range(0,len(unit_code_list))):
        total_num+=len(unit_code_list[i])
        for unit_code in unit_code_list[i]:
            unit_test = unit_code['unit_test']
            result = unit_code['result']
            replace_part = extract_functions(unit_test)
            unit_code = unit_test.replace(replace_part,code_list[i])
            test_result = unit_test_utils.get_subroutine_result(unit_code)
            if test_result==result:
                correct_num+=1
    return pass_at_k(total_num,correct_num,1)
def main():
    benchmark_path = "./alpaca_function/function_benchmark_alpaca.jsonl"
    test_path = ""
    benchmark_list = utils.read_json(benchmark_path)
    benchmark_code = [data['test_case'] for data in benchmark_list]
    test_list = utils.read_json(test_path)
    test_list = [extract_fortran(data['code']) for data in test_list]
    pass_1 = run_code(benchmark_code,test_list)
    print(f"pass@1:{pass_1},compile_pass:{get_compile_pass(test_list)}")
if __name__ == "__main__":
    main()

