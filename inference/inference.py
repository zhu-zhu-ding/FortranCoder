import os

# 设置CUDA_VISIBLE_DEVICES环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
from peft import PeftModel, LoraConfig, get_peft_model

from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
sys.path.append('/home/data1/wpd1/FortranCoder')
from utils import (
    read_json,
    save_json,
)
from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--data_type", type=str, required=True)
    args = parser.parse_args()


    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
    # # 加载lora模型
    if args.lora_path!='':
        model = PeftModel.from_pretrained(model, args.lora_path)
        model = model.merge_and_unload()


    #inference函数，可以直接调用
    def inference(instruction):
        messages=[
                { 'role': 'user', 'content': instruction}
        ]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
        
        # tokenizer.eos_token_id is the id of <|EOT|> token
        outputs = model.generate(inputs, max_new_tokens=512,temperature=0.2,do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
        
        return tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)


    test_data = read_json(args.data_path,True)
    test_data = test_data[:3]
    for item in tqdm(test_data):
        if args.data_type =='HumanEval':
            item['completion'] = inference(item['prompt'])
        else:
            item['answer'] = inference(item['instruction'])

        save_json(args.save_path,test_data)


if __name__ == "__main__":
    main()