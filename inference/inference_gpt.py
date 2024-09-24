import os

# 设置CUDA_VISIBLE_DEVICES环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
import argparse
from gpt_api_base import call_openai
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--data_type", type=str, required=True)
    args = parser.parse_args()


    def inference(instruction):
        messages=[
                { 'role': 'user', 'content': instruction}
        ]
        return call_openai(messages,temperature=0.2)


    test_data = read_json(args.data_path,True)
    test_data = test_data[:3]
    for item in tqdm(test_data):
        if args.data_type =='HumanEval':
            item['completion'] = inference(item['prompt'])
        else:
            item['answer'] = inference(item['instruction'])

        save_json(args.save_path,test_data)
