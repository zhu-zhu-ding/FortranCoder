import json
import os
import concurrent.futures
import threading
from datasketch import MinHash,MinHashLSH
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
from utils import (
    read_json,
    save_json,
    compile_test
)
function_source_data_path = ''
subroutine_source_data_path = ''

function_source_data = read_json(function_source_data_path,True)
subroutine_source_data = read_json(subroutine_source_data_path,True)


function_source_data = [data['code'] for data in function_source_data]
subroutine_source_data = [data['code'] for data in subroutine_source_data]

def deduplication(fortran_list,threshold=0.7):
    """
    :param fortran_list: Fortran function list
    :return: results after removing duplicates
    """
    index = 0
    result_code=[]
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    for  code in tqdm(fortran_list, desc="dedup processing"):
        minhash = MinHash(num_perm=128)
        minhash_query = MinHash(num_perm=128)
        for word in code.split():
            minhash_query.update(word.encode('utf-8'))
        is_repeat = lsh.query(minhash_query)
        if not is_repeat:
            for word in code.split():
                minhash.update(word.encode('utf-8'))
            lsh.insert(str(index), minhash)
            index += 1
            result_code.append(code)
    return result_code


def is_code_meaningful(fortran_code, min_threshold=10 ,max_threshold = 30):

    # 去掉空行和仅包含注释的行
    lines = [line for line in fortran_code.split('\n') if line.strip() and not line.strip().startswith('!')]
    
    # 判断行数是否超过阈值
    return len(lines) >= min_threshold and len(lines) <= max_threshold

def do_deduplication(function_path,subroutine_path,threshold=0.7):
    function_list = deduplication(function_source_data,threshold)
    subroutine_list = deduplication(subroutine_source_data,threshold)
    print(f"len dedup_function: {len(function_list)}, len dedup_subroutine:{len(subroutine_list)}")
def do_check_meaningful(function_path,subroutine_path):
    function_list = read_json(function_path)
    subroutine_list = read_json(subroutine_path)
    function_list = [data for data in tqdm(function_list) if is_code_meaningful(data)]
    save_json("",function_list)
    subroutine_list = [data for data in tqdm(subroutine_list) if is_code_meaningful(data)]
    save_json("",subroutine_list)
    print(f"len check_function: {len(function_list)}, len check_subroutine:{len(subroutine_list)}")
def filter_valid_fortran_codes(codes, max_workers=4):
    valid_codes = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_code = {executor.submit(compile_test, code): code for code in codes}
        
        for future in as_completed(future_to_code):
            code = future_to_code[future]
            try:
                if future.result():
                    valid_codes.append(code)
            except Exception as e:
                print(f"Error processing code: {e}")
    
    return valid_codes
def do_check_compile(function_path,subroutine_path):
    function_list = read_json(function_path)
    subroutine_list = read_json(subroutine_path)

    function_list = filter_valid_fortran_codes(function_list)
    save_json("",function_list)

    subroutine_list = filter_valid_fortran_codes(subroutine_list)

    save_json("",subroutine_list)
    print(f"len compile_test_function: {len(function_list)}, len compile_test_subroutine:{len(subroutine_list)}")
def main():
    function_list = deduplication(function_source_data,0.7)
    subroutine_list = deduplication(subroutine_source_data,0.7)
    function_list = [data for data in tqdm(function_list) if is_code_meaningful(data)]
    subroutine_list = [data for data in tqdm(subroutine_list) if is_code_meaningful(data)]
    function_list = filter_valid_fortran_codes(function_list)
    subroutine_list = filter_valid_fortran_codes(subroutine_list)
    save_json("",function_list)
    save_json("",subroutine_list)
    print(f"len compile_test_function: {len(function_list)}, len compile_test_subroutine:{len(subroutine_list)}")
if __name__ == "__main__":
    main()

