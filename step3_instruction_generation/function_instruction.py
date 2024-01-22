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
read_path = '../step2_test_case_generation/data/step2_function_result.jsonl'
save_path = './data/step3_function_data.jsonl'
def set_prompt(type_in_num,type_in,type_out_num,type_out):
    return f"""As a Fortran expert, you are required to provide an instruction for the following Fortran function. You must comply with the following requirements:
1. You must understand the code correctly and briefly summarize the function of the function in the instructions.
2. You must reflect the features of Fortran function in the instruction, for example, Fortran, function and other keywords appear in the instruction.
3. The generated instructions use the imperative tone.
4. You must specify the number, type and function of input and output parameters in the instruction. The number of input parameters in this code is {type_in_num}, the test_data type is {type_in}, the number of output parameters is {type_out_num}, the test_data type is {type_out}.
5. You must specify the name of the function in the instruction.
    """

def function_message(type_in_num,type_in,type_out_num,type_out,fortran_code):
    answer_ex1 = "Give me a Fortran function code named add_numbers to compute the sum of two floating numbers.The number of input arguments is 2, the first parameter is a floating number, the type is real; The second argument is a floating number,the type is real. The number of output arguments is 1, the argument is the sum of two floating numbers and the type is real."
    code_ex1 = """function add_numbers(a, b) result(sum)
              implicit none
              real :: a, b, sum
              sum = a + b
            end function add_numbers
            """
    message = [
        {"role": "system", "content": code_ex1},
        {"role": "user", "content": set_prompt(2,["real","real"],1,["real"])},
        {"role": "assistant", "content": answer_ex1},
        {"role": "system", "content": fortran_code},
        {"role": "user", "content": set_prompt(type_in_num,type_in,type_out_num,type_out)}
    ]
    return message

def main():
    data_list = utils.read_json(read_path)
    data_list = [data for data in data_list if len(data['test_case'])==6]
    save_list = []
    for data in tqdm(data_list):
        instruction_message = function_message(data['param_info']['type_in_num'],
                              data['param_info']['type_in'],
                              data['param_info']['type_out_num'],
                              data['param_info']['type_out'],
                              data['code'])
        instructions = gpt_api_base.call_openai(message=instruction_message,n=3)
        for instruction in instructions:
            # filter
            if unit_test_utils.get_function_name(data['code']) in instruction:
                save_list.append({'instruction':instruction,'param_info':data['param_info'],'test_case':data['test_case'],'code':data['code']})
                break
        if len(save_list)%100==0:
            utils.save_json(save_path = save_path,save_list=save_list)
    utils.save_json(save_path=save_path,save_list=save_list)
if __name__ == '__main__':
    main()

