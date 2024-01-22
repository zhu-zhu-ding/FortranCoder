import os
import transformers
import fire
import copy
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import Trainer, TrainingArguments, HfArgumentParser, set_seed
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)
import torch
from dataclasses import field, fields, dataclass
import bitsandbytes as bnb

from alpaca_dataset import load_alpaca_dataset

#model_path
model_path = ''
#finetune_path
data_path = ''
#save_lora
output_dir = ''
def main():
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        load_in_8bit=True,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
    )
    model = prepare_model_for_int8_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules = ["attn.c_proj", "attn.c_attn"]
    )

    model = get_peft_model(model, lora_config)
    ############# prepare data ###########
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    data = load_alpaca_dataset(data_path, tokenizer, max_len=512)
    all_data = data.train_test_split(
            train_size=1, shuffle=True, seed=1234
        )
    train_data = all_data["train"].shuffle(seed=1234)
    eval_data = all_data["test"].shuffle(seed=1234)

    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=4,
            warmup_steps=100,
            num_train_epochs=3,
            learning_rate=3e-4,
            fp16=False,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=200,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False,
            group_by_length=False,
            report_to="wandb"
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer,
                                                          pad_to_multiple_of=8,
                                                          return_tensors="pt",
                                                          padding=True),
    )
    trainer.train(resume_from_checkpoint=False)
    model.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(main)
