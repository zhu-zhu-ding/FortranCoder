import re
from copy import deepcopy
import random
import time
import json
from tqdm import tqdm

import sys

sys.path.append('/home/wpd/back_fortran_subroutine')
import gpt_api_base
import utils
import unit_test_utils
read_function_path = 'data/parse_function_result.jsonl'
save_function_path = 'data/step2_function_result.jsonl'
def set_function_prompt():
    return f"""As a Fortran expert, you are required to complete the test code for the following Fortran function to obtain the output.Fortran function test code as follows:
            '''
        program main
            use test_function
        !to complete the test code
        end program main 
        '''
        You must comply with the following requirements:
        1.You only need to generate a test case but you have to make sure that the test case is correct , comprehensive and complex.
        2.You don't need loop prints.You only need to print the output variable, which is of type intent(out), and you can't print anything else.
        """
def gen_function_prompt(fortran_code):
    answer_ex = f"""
    program main
      use test_function
      implicit none
      real :: total
      integer :: m = 2
      real, dimension(2) :: arr = [-1.042, 2.421]
      total = sum_array(m,arr)
      print *, total
    end program main
        """
    code_ex = """  function sum_array(m,arr) result(total)
          implicit none
          integer,intent(in):: m
          real, dimension(m), intent(in) :: arr
          real :: total
          integer :: i

          total = 0.0
          do i = 1, size(arr)
            total = total + arr(i)
          end do
        end function sum_array
                            """
    message = [
        {"role": "system", "content": code_ex},
        {"role": "user", "content": set_function_prompt()},
        {"role": "assistant", "content": answer_ex},
        {"role": "system", "content": fortran_code},
        {"role": "user", "content": set_function_prompt()}
    ]
    return message
def extract_fortran(fortran_code):
    pattern = r"```fortran(.*?)```"
    matches = re.findall(pattern, fortran_code, re.DOTALL)
    if matches:
        extracted_code = matches[0]
        return extracted_code
    else:
        return fortran_code
def result_filter(result):
    ban_list = ['NaN','Infinity','-Infinity']
    temp_result = []
    for data in result:
        f = True
        for item in data['result']:
            for ban in ban_list:
                if ban in item:
                    f=False
        if f:
            temp_result.append(data)
    count_list = {str(data['result']):0 for data in temp_result}
    last_result = []
    if len(temp_result)>=1:
        for data in temp_result:
            if count_list[str(data['result'])]==0:
                count_list[str(data['result'])]+=1
                last_result.append(data)
            else:
                continue
    return last_result
def code_filter(data):
    test_list = data['test_case']
    f = True
    for test in test_list:
        result = test['result']
        code = test['unit_test']
        test_result = unit_test_utils.get_function_result(code)
        if test_result == result:
            f = True
        else:
            f = False
    if f:
        return True
    else:
        return False
def generate_test_case(data):
    n=3
    code = data['code']
    test_code_part = f"""
                        module test_function
                        contains
                        {code}
                        end module test_function
                        """
    result = []
    while(n>0):
        n-=1
        test_code_list = gpt_api_base.call_openai(message=gen_function_prompt(code),temperature=0.8,n=3)
        for test_code in test_code_list:
            test = test_code_part+extract_fortran(test_code)
            temp_result = unit_test_utils.get_function_result(test)
            if temp_result!=None:
                result.append({'unit_test':test,'result':temp_result})
        if len(result)!=0:
            result = result_filter(result)
        if len(result)>=5:
            break

    return result

def main():
    data_list = utils.read_json(read_path=read_function_path)
    result_list = []
    for i,data in tqdm(enumerate(data_list),total=len(data_list),desc='call gpt processing'):
        result_list.append({'id':i,
                    'param_info':data['param_info'],
                    'test_case':generate_test_case(data=data),
                    'code':data['code']})
        if len(result_list)%100==0:
            result_list = [data for data in result_list if len(data['test_case']) != 0 and code_filter(data)]
            utils.save_json(save_path = save_function_path,save_list=result_list)
    result_list = [data for data in result_list if len(data['test_case'])!=0 and code_filter(data)]
    result_list = [{'id':i,
                    'param_info':data['param_info'],
                    'test_case':data['test_case'],
                    'code':data['code']} for i,data in enumerate(result_list)]
    utils.save_json(save_path = save_function_path,save_list=result_list)
if __name__ == "__main__":
    main()