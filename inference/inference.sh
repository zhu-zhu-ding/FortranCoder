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