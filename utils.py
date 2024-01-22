import json
import os


def read_json(read_path):
    function_list = []
    try:
        with open(read_path, 'r',encoding="utf-8") as jsonl_file:
            for line in jsonl_file:
                json_obj = json.loads(line)
                function_list.append(json_obj)
        return function_list
    except Exception as e:
        print(f"read json_path {read_path} exception:{e}")
        return None

def save_json(save_path,save_list):
    try:
        with open(save_path, 'w', encoding="utf-8") as jsonl_file:
            for save_item in save_list:
                json_string = json.dumps(save_item) + '\n'
                jsonl_file.write(json_string)
            print(f"save data to {save_path}")
    except Exception as e:
        print(f"save json_path {save_path} exception:{e}")
