import json
import utils

function_ft_path = './function/function_ft.jsonl'
subroutine_ft_path = './subroutine/subroutine_ft.jsonl'
save_ft_path = './alpaca/all_ft_alpaca.jsonl'

function_ft = utils.read_json(function_ft_path)
last_result = [{"instruction":data["instruction"],"input":"","output":data['code']} for data in function_ft]
subroutine_ft = utils.read_json(subroutine_ft_path)
save_ft_2 = [{"instruction":data["instruction"],"input":"","output":data['code']} for data in subroutine_ft]
last_result.extend(save_ft_2)

with open(save_ft_path, 'w', encoding="utf-8") as jsonl_file:
    jsonl_file.write(json.dumps(last_result))
print(f"数据已保存到 {save_ft_path}")