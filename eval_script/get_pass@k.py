from tqdm import tqdm

import sys
sys.path.append('/home/data1/wpd1/Fortran')
import utils
import unit_test_utils
import argparse

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
def extract_fortran(fortran_code):
    pattern = r"```fortran(.*?)```"
    matches = re.findall(pattern, fortran_code, re.DOTALL)
    if matches:
        extracted_code = matches[0]
        return extracted_code
    else:
        return fortran_code
def extract_cl_fortran(fortran_code):
    pattern = r"```(.*?)```"
    matches = re.findall(pattern, fortran_code, re.DOTALL)
    if matches:
        extracted_code = matches[0]
        return extracted_code
    else:
        return fortran_code
def run_function_code(data_list):
    correct_num = 0
    total_num = 0
    for data in tqdm(data_list):
        total_num+=1
        unit_test = data['test_case']['unit_test']
        result =data['test_case']['result']
        test_function  = extract_fortran(data['answer'])
        test_code = f'''
    module test_function
       contains
      {test_function}
    end module test_function
{unit_test}
    '''
        # test_code = f'''module test_function
        # contains
        # {test_function}
        # module test_function
        # {unit_test}'''
        test_result = unit_test_utils.get_function_result(test_code)
        # print(test_result,result)
        if test_result==result:
            correct_num+=1
        else:
            try:
                if all(are_strings_equal(str1, str2) for str1, str2 in zip(test_result, result)):
                    correct_num+=1
            except:
                continue
    return pass_at_k(total_num,correct_num,1)
def run_subroutine_code(data_list):
    correct_num = 0
    total_num = 0
    for data in tqdm(data_list):
        total_num+=1
        unit_test = data['test_case']['unit_test']
        result =data['test_case']['result']
        test_function  = extract_fortran(data['answer'])
        test_code = f'''
    module test_subroutine
       contains
      {test_function}
    end module test_subroutine
{unit_test}
    '''
        # test_code = f'''module test_function
        # contains
        # {test_function}
        # module test_function
        # {unit_test}'''
        test_result = unit_test_utils.get_subroutine_result(test_code)
        # print(test_result,result)
        if test_result==result:
            correct_num+=1
        
        else:
            try:
                if all(are_strings_equal(str1, str2) for str1, str2 in zip(test_result, result)):
                    correct_num+=1
            except:
                continue
    return pass_at_k(total_num,correct_num,1)
import numpy as np
import re

def are_strings_equal(str1, str2, tolerance=1e-9):
    # 去除换行符并按空格分割
    numbers1 = np.array([float(num) for num in re.sub(r'[\n(),]', ' ', str1).split()])
    numbers2 = np.array([float(num) for num in re.sub(r'[\n(),]', ' ', str2).split()])
    # 比较两个数组的长度
    if len(numbers1) != len(numbers2):
        return False
    # 比较每个元素，允许误差 tolerance
    return np.allclose(numbers1, numbers2, atol=tolerance)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--data_type", type=str, required=True)
    args = parser.parse_args()

    data_list = utils.read_json(args.data_path)

    if args.data_type=='function':
        pass_1 = run_function_code(data_list)
    else:
        pass_1 = run_subroutine_code(data_list)
    test_list = [extract_fortran(data['answer']) for data in data_list]
    print(f"pass@1:{pass_1},compile_pass:{get_compile_pass(test_list)}")
if __name__ == "__main__":
    main()

