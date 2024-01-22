import json
import utils

fortran_list = []
last_result =[]
read_benchmark_path = './subroutine/subroutine_benchmark.jsonl'
save_benchmark_path = './alpaca_subroutine/subroutine_benchmark_alpaca.jsonl'
read_ft_path = './subroutine/subroutine_ft.jsonl'
save_ft_path = './alpaca_subroutine/subroutine_ft_alpaca.jsonl'

read_benchmark = utils.read_json(read_benchmark_path)
save_benchmark = [{"instruction":data["instruction"],"input":"","output":data['code']} for data in read_benchmark]
utils.save_json(save_path=save_benchmark_path,save_list=save_benchmark)

read_ft = utils.read_json(read_ft_path)
save_ft = [{"instruction":data["instruction"],"input":"","output":data['code']} for data in read_ft]
with open(save_ft_path, 'w', encoding="utf-8") as jsonl_file:
    jsonl_file.write(json.dumps(save_ft))
print(f"数据已保存到 {save_ft_path}")