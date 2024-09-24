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
from gpt_api_base import call_openai
from tqdm import tqdm

function_test_data_path = '/home/data1/wpd1/Fortran/test/function_filter_data/epoch1/CodeLlama-7b-Instruct-hf_lora.jsonl'
function_test_data = read_json(function_test_data_path,True)
for item in function_test_data:
    item['answer'] = item['answer'].replace('<|EOT|>','')
save_json("/home/data1/wpd1/Fortran/test/function_filter_data/epoch1/CodeLlama-7b-Instruct-hf_lora_1.jsonl",function_test_data,is_list = True)
