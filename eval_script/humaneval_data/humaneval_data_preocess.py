import os

# 设置CUDA_VISIBLE_DEVICES环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
from peft import PeftModel, LoraConfig, get_peft_model

from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
sys.path.append('/home/data1/wpd1/Fortran')
from utils import (
    read_json,
    save_json,
    compile_test
)
import re
from gpt_api_base import call_openai
from tqdm import tqdm

function_test_data_path = '/home/data1/wpd1/Fortran/test/humaneval_data/CodeLlama-7b-Instruct-hf/CodeLlama-7b-Instruct-hf_lora.jsonl'
def extract_fortran(fortran_code):
    pattern = r"```fortran(.*?)```"
    pattern_1 = r"```(.*?)```"
    matches = re.findall(pattern, fortran_code, re.DOTALL)
    matches_1 = re.findall(pattern_1, fortran_code, re.DOTALL)
    if matches:
        extracted_code = matches[0]
        return extracted_code
    elif matches_1:
        return matches_1[0]
    else:
        return fortran_code
function_test_data = read_json(function_test_data_path,True)
for item in function_test_data:
    test_code = f'''
    module test_module
       contains
      {extract_fortran(item['completion'])}
    end module test_module
    '''
    item['completion'] = test_code
save_json("/home/data1/wpd1/Fortran/test/humaneval_data/CodeLlama-7b-Instruct-hf/CodeLlama-7b-Instruct-hf_lora_1.jsonl",function_test_data,is_list = False)
