import json
import os
import subprocess
import threading


def read_json(data_path,is_list=True):
    if is_list:
        try:
            return json.load(open(data_path, 'r',encoding="utf-8"))
        except Exception as e:
            print(f"read json_path {data_path} exception:{e}")
            return None
    else:
        try:
            return [json.loads(line) for line in open(data_path, 'r',encoding="utf-8")]
        except Exception as e:
            print(f"read json_path {data_path} exception:{e}")
            return None

def save_json(data_path,data_list,is_list=True):
    if is_list:
        try:
            open(data_path, 'w', encoding="utf-8",).write(json.dumps(data_list,indent=4))
        except Exception as e:
            print(f"save json_path {data_path} exception:{e}")
            return None
    else:
        try:
            with open(data_path, 'w', encoding="utf-8") as jsonl_file:
                for save_item in data_list:
                    jsonl_file.write(json.dumps(save_item) + '\n')
        except Exception as e:
            print(f"save json_path {data_path} exception:{e}")
            return None

def compile_test(code):
    # 测试代码
    test_code = f'''
    module test
       contains
        {code}
    end module test
    program main
      use test
      implicit none
    end program main
    '''
    
    code_file = f'temp{threading.get_ident()}.f90'
    exe_file = f'temp{threading.get_ident()}'
    
    with open(code_file, 'w',encoding="utf-8") as file:
        file.write(test_code)
    try:
        subprocess.check_output(['gfortran', '-o', exe_file, code_file], stderr=subprocess.STDOUT)
        return True
    except Exception as e:
        return False
    finally:
        # 清理生成的文件
        if os.path.exists(code_file):
            os.remove(code_file)
        if os.path.exists(exe_file):
            os.remove(exe_file)
        if os.path.exists(f'{exe_file}.exe'):
            os.remove(f'{exe_file}.exe')
