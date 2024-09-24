import torch
from peft import PeftModel, LoraConfig, get_peft_model

from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
from utils import (
    read_json,
    save_json,
    compile_test
)
from gpt_api_base import call_openai
from tqdm import tqdm


def build_instruction_prompt(code: str):
    prompt_template = f'''### Instruction:
    As a Fortran expert, you are exceptionally skilled at crafting an programming instruction for the following function-level code.
    Please understand the following Fortran code snippet to create a high-quality programming instruction for this Fortran code snippet.
    **Please distinguish between Fortran's two function types, function and subroutine, and reflect them in instruction.**
    {code}
    ### Response:
    '''
    return prompt_template

function_answer_ex1 = "Give me a Fortran function code named add_numbers to compute the sum of two floating numbers.The number of input arguments is 2 and the test_data type is <real,real>.The number of output arguments is 1 and the test_data type is <real>."
function_code_ex1 = """function add_numbers(a, b) result(sum)
              implicit none
              real :: a, b, sum
              sum = a + b
            end function add_numbers"""
subroutine_answer_ex1 = "Give me a Fortran subroutine code called add_numbers to compute the sum of an array. The number of input parameters is 2, the first parameter is the size of the array, the type is integer; The second argument is an array of type real. The number of output arguments is 1, the argument is the sum of the array, and the type is real."
subroutine_code_ex1 = """subroutine sum_array(m, arr, total)
    implicit none
    integer, intent(in) :: m
    real, dimension(m), intent(in) :: arr
    real, intent(out) :: total
    integer :: i
      total = 0.0
      do i = 1, size(arr)
        total = total + arr(i)
      end do
    end subroutine sum_array"""
#inference函数，可以直接调用
def inference(code,type= 'function'):
    if type == 'function':
        messages=[
            { 'role': 'user', 'content': build_instruction_prompt(function_code_ex1)},
            { 'role': 'assistant', 'content': function_answer_ex1},
            { 'role': 'user', 'content': build_instruction_prompt(code)}
        ]
    else:
        messages=[
            { 'role': 'user', 'content': build_instruction_prompt(subroutine_code_ex1)},
            { 'role': 'assistant', 'content': subroutine_answer_ex1},
            { 'role': 'user', 'content': build_instruction_prompt(code)}
        ]
    
    return call_openai(messages,temperature=0)

def process_code(code, type='function'):
    instruction = inference(code, type)
    format_code = f"""```fortran\n{code}\n```"""
    return {'instruction': instruction, 'output': format_code,'code':code}

function_task = read_json('')
function_task_result = []

for code in tqdm(function_task):
    function_task_result.append(process_code(code,'function'))
    save_json("",function_task_result)

scientific_task_subroutine = read_json('')
# scientific_task_subroutine = scientific_task_subroutine[:5]
scientific_task_subroutine_result = []
for code in tqdm(scientific_task_subroutine):
    scientific_task_subroutine_result.append(process_code(code,'subroutine'))
    save_json("",scientific_task_subroutine_result)