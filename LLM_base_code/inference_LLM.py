import os
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from datasets import load_dataset
from peft import PeftModel
import os
import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
import json
from tqdm import tqdm
from peft import PeftModel, LoraConfig, get_peft_model
from load_model import load_model
from peft import PeftModel
from transformers import GenerationConfig
from alpaca_dataset import load_alpaca_dataset


# read_inference_data_path
data_path = ""
# model_path
model_path = ''
# lora_path
lora_path = ""
# save_inference_data_path
save_path = ""
model, tokenizer = load_model(model_path,False)

# use lora
model = PeftModel.from_pretrained(model, lora_path)

# model.train(False)
# model.eval()
device = model.device
generation_config = GenerationConfig(
    temperature=0,
    top_p=1,
    max_new_tokens=2048,  # max_length=max_new_tokens+input_sequence
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id
)

def generate_prompt(instruction, input=None):
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""

def generate_model_prompt(data_point):
    full_prompt = ""
    instruction = data_point["instruction"]
    full_prompt = generate_prompt(instruction)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    return full_prompt,inputs.input_ids


def load_test_data(data_path):
    instructions=[]
    with open(data_path, "r") as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            instruction,inputs = generate_model_prompt(data)
            instructions.append((instruction,inputs))
    return instructions


def save_result_data(last_result,save_path):
    with open(save_path, "w", encoding="utf-8") as jsonl_file:
        for item in last_result:
            assert len(item) == 2
            temp = {}
            temp["instruction"] = item[0]
            temp["code"] = item[1]
            json_string = json.dumps(temp) + "\n"
            jsonl_file.write(json_string)
    print(f"数据已保存到{save_path}")


data = load_test_data(data_path)
last_result = []
for i, data_point in tqdm(enumerate(data), total=len(data), desc="Processing"):

    generate_ids = model.generate(input_ids=data_point[1], generation_config=generation_config)
    output = tokenizer.decode(
        generate_ids[0], skip_special_tokens=True
    )
    last_result.append((data_point[0],output.split("### Response:")[1].strip()))

save_result_data(last_result,save_path)