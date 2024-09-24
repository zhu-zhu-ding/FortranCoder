import sys
sys.path.append('/home/data1/wpd1/Fortran')
from utils import (
    read_json,
    save_json,
)
from gpt_api_base import call_openai
from tqdm import tqdm


def build_instruction_prompt(code: str,instruction):
    prompt_template = f'''You are an expert in Fortran programming, please provide a Fortran code snippet according to the following instruction and code.
{instruction}
{code}.'''
    return prompt_template

evol_list = [
    'If the code function type is function, please rewrite it as subroutine. If the function type is subroutine, please rewrite it as function.',
    'Propose higher time and space complexity requirements for the following code.',
    'Provide an implementation with more challenging and complex functions or subroutines relevant to scientific computing.',
    'Provide an implementation with more challenging and complex functions or subroutines relevant to general programming tasks.'
]

def inference(code,instruction):
    messages=[
            { 'role': 'user', 'content': build_instruction_prompt(code,instruction)}
    ]
    return call_openai(messages,temperature=0.8)


function = read_json('')
# function = function[:3]
function_result = []
for data in tqdm(function):
    code = data
    for instruction in evol_list:
        function_result.append({'instruction':build_instruction_prompt(code,instruction),'output':inference(code,instruction)})
    save_json("",function_result)