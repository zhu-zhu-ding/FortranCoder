# Data Overview
## Train Data
```bash
FortranCoder/lora_model/train_Evol_Code/train_Evol_Code.json
```

## Test Data
```bash
FortranCoder/FortranEval
```

# Fine-Tuning
The script is in the fintune dir.

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

