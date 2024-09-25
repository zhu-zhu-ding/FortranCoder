# About
+ **<font style="color:rgb(31, 35, 40);">Fortrancoder</font>**<font style="color:rgb(31, 35, 40);"> is a model empowered by </font>**<font style="color:rgb(31, 35, 40);">Evol-Code, </font>**<font style="color:rgb(31, 35, 40);">a novel approach to improve Fortran Programming for LLMs.</font>
+ **<font style="color:rgb(31, 35, 40);">Evol-Code</font>**** **extends the diversity of instructions with data collected from real programming scenarios to improve the overall programming capabilities of LLM.
+ **FortranEval** is a benchmark dataset that comprehensively evaluates the Fortran programming capabilities of LLM. It includes both **function** and **subroutine**, and comprehensive programming tasks including scientific computing and general programming tasks.

💫Important !!!!!

🏅 **<font style="color:rgb(31, 35, 40);">Fortrancoder-DS-6.7B</font>**<font style="color:rgb(31, 35, 40);"> outperforms </font>**<font style="color:rgb(31, 35, 40);">gpt-3.5-turbo-1106</font>**<font style="color:rgb(31, 35, 40);"> and </font>**<font style="color:rgb(31, 35, 40);">DeepSeek-Coder-6.7B-Instruct</font>**<font style="color:rgb(31, 35, 40);"> ( base model ) on FortranEval!（32.5% vs [29.5% and 27.4%] on pass@1 and 80.8% vs [72.6% and 70.9%]）！</font>

![](https://cdn.nlark.com/yuque/0/2024/png/38861830/1727250691535-0f8c97b0-4250-4b56-aea5-2b75fddf3952.png)

# 🤖Models
| **<font style="color:rgb(31, 35, 40);">Model</font>** | **<font style="color:rgb(31, 35, 40);">Checkpoint</font>** | **<font style="color:rgb(31, 35, 40);">Size</font>** | **<font style="color:rgb(31, 35, 40);">pass@1</font>** | **<font style="color:rgb(31, 35, 40);">pass_c@1</font>** | **<font style="color:rgb(31, 35, 40);">License</font>** |
| --- | --- | --- | --- | --- | --- |
| FortranCoder-DS-6.7B | <font style="color:rgb(31, 35, 40);">🤗</font><font style="color:rgb(31, 35, 40);"> </font>[HF_Link](https://huggingface.co/zzzzd/FortranCoder-DS-6.7B/tree/main) | 6.7B | 32.5(<font style="color:rgb(31, 35, 40);">27.4)</font> | 80.8(<font style="color:rgb(31, 35, 40);">70.9)</font> | [DeepSeek](https://github.com/deepseek-ai/DeepSeek-Coder/blob/main/LICENSE-MODEL) |


# 📚<font style="color:rgb(31, 35, 40);">Dataset</font>
[**FortranEval**](https://github.com/zhu-zhu-ding/FortranCoder/tree/main/FortranEval)**:** **The first benchmark dataset to evaluate LLM's Fortran programming capabilities.**

[**Evol-Code-Fortran**](https://github.com/zhu-zhu-ding/FortranCoder/blob/main/finetune/train_Evol_Code.json)**: Fortran instruction fine-tuning data generated using the Evol-Code method.**

# Fine-Tuning
<font style="color:rgb(31, 35, 40);">The script supports the training with </font>[DeepSpeed](https://github.com/microsoft/DeepSpeed)<font style="color:rgb(31, 35, 40);">. You need install required packages by:</font>

```bash
pip install -r requirements.txt
```

The script **<font style="color:rgb(31, 35, 40);">finetune_deepseekcoder.py</font>**<font style="color:rgb(31, 35, 40);"> </font>we provide is as [DeepSeek](https://github.com/zhu-zhu-ding/FortranCoder/tree/main/finetune).

```bash
DATA_PATH="{your_path}"
OUTPUT_PATH="{your_path}"
MODEL_PATH="{your_path}"

#wandb login
export CUDA_VISIBLE_DEVICES=1,2



deepspeed --num_gpus 2 --master_port 6002 finetune/lora_deepseekcoder.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 2 \
    --model_max_length 2048 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps 100 \
    --save_total_limit 100 \
    --learning_rate 1e-5 \
    --warmup_steps 10 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --report_to "wandb" \
    --deepspeed finetune/configs/ds_config_zero3.json \
    --bf16 True
```

# Inference
The script is in the inference dir.

```bash
python inference.py \
--data_path {your_path}/FortranCoder/FortranEval/FortranEval_base_function.jsonl \
--model_path {your_path}/DeepSeek-Coder/model/deepseek-coder-6.7b-instruct \
--lora_path /{your_path}/FortranCoder/lora_model/train_Evol_Code/model/deepseek-coder-6.7b-instruct \
--save_path {your_path}/FortranCoder/inference/1.jsonl \
--data_type fortran   #fortran or HumanEval


# python inference_gpt.py \
# --data_path {your_path}/FortranCoder/FortranEval/FortranEval_base_function.jsonl \
# --save_path {your_path}/FortranCoder/inference/1.jsonl \
# --data_type fortran   #fortran or HumanEval
```

## 🚤<font style="color:rgb(31, 35, 40);">Quick Start</font>
```bash
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()

messages=[
    { 'role': 'user', 'content': "write a quick sort algorithm in python."}
]

inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
# tokenizer.eos_token_id is the id of <|EOT|> token
outputs = model.generate(inputs, max_new_tokens=512, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))
```

