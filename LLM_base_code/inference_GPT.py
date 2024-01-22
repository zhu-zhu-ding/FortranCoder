from tqdm import tqdm

import utils
import gpt_api_base

read_path = './data/function_benchmark_alpaca.jsonl'
save_path = ''
test_list = utils.read_json(read_path)
def gen_message(instruction):
    message = [
        {"role": "user", "content": instruction}
    ]
    return message
result_list = []
for data in tqdm(test_list):
    message = gen_message(data['instruction'])
    result = gpt_api_base.call_openai(message = message,n=1,temperature=0)
    result_list.append({'instruction':data['instruction'],'code':result})
utils.save_json(save_path=save_path,save_list=result_list)
