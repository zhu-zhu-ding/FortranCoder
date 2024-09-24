
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