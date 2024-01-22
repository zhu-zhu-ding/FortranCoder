import re
import subprocess
import os
import threading
from datasketch import MinHash,MinHashLSH
from tqdm import tqdm
import utils
import unit_test_utils

read_path = '../source_code/fortran_example.jsonl'
function_save_path = './data/step1_function_result.jsonl'
subroutine_save_path = './data/step1_subroutine_result.jsonl'

#deduplication function
def deduplication(fortran_list,threshold=0.7):
    """
    :param fortran_list: Fortran function list
    :return: results after removing duplicates
    """
    index = 0
    result_code=[]
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    for code in tqdm(fortran_list, desc="dedup processing"):
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

def check_code(code,type ='function'):
    """
    判断给定的 Fortran语法正确有输入输出：
    1.指定位置包含result关键字
    2.非注释行数 - 变量定义行大于等于2
    :param code: Fortran 代码字符串
    :return: 如果满足条件，返回 True；,否则返回 False。
    """
    lines = code.split('\n')    # 将代码字符串按行分割成列表
    count = 0                   # 记录非注释行数
    define_count = 0            # 记录变量定义行数
    has_return = False          # 记录是否有返回值
    function_name = unit_test_utils.get_function_name(code)
    for i,line in enumerate(lines):
        # 使用正则表达式匹配注释行
        if re.match(r'^\s*!', line):
            continue    # 如果是注释行，则跳过本次循环
        # 剔除行内注释部分
        line = re.sub(r'!.+', '', line)
        # 判断剔除注释后的代码行是否为空
        if not line.strip():
            continue    # 如果是空行，则跳过本次循环
        count += 1      # 非注释行数加一
        define_count_str = re.escape('::')
        if re.search(define_count_str,line):
            define_count+=1
        if type == 'function':
            if ((i==0 and re.search(r'\bresult\b', line)) or
                    (i!=0 and i !=len(lines)-1 and re.search(fr'\b{function_name}\b', line))):
                has_return = True
        elif type == 'subroutine':
            if (i != 0 and ('intent(out)' in line)):
                has_return = True
    if has_return and (count-define_count) >2:
        return True
    else:
        return False

def main():
    data_list = utils.read_json(read_path=read_path)
    # print(len(data_list))
    # data_list = data_list[:1000]
    function_list = []
    subroutine_list = []
    for data in data_list:
        function_list.extend(unit_test_utils.extract_function(data['text']))
        subroutine_list.extend(unit_test_utils.extract_subroutine(data['text']))
    function_list = deduplication(function_list,threshold=0.7)
    subroutine_list = deduplication(subroutine_list,threshold=0.7)

    # function_list = [{'id':i,'function':save_item}
    #              for i,save_item in enumerate(function_list)]
    # utils.save_json(save_path='./data/dedup_function.jsonl', save_list=function_list)
    # subroutine_list = [{'id':i,'subroutine':save_item}
    #              for i,save_item in enumerate(subroutine_list)]
    # utils.save_json(save_path='./data/dedup_subroutine.jsonl', save_list=subroutine_list)

    function_list = [function for function in tqdm(function_list, desc="check processing")
                   if (unit_test_utils.check_iscompile(function) and check_code(function,type='function'))]
    subroutine_list = [subroutine for subroutine in tqdm(subroutine_list, desc="check processing")
                   if (unit_test_utils.check_iscompile(subroutine) and check_code(subroutine,type='subroutine'))]

    function_list = [{'id':i,'function':save_item}
                 for i,save_item in enumerate(function_list)]
    subroutine_list = [{'id':i,'subroutine':save_item}
                 for i,save_item in enumerate(subroutine_list)]
    utils.save_json(save_path=subroutine_save_path,save_list=subroutine_list)
    utils.save_json(save_path=function_save_path,save_list=function_list)

if __name__ == "__main__":
    main()

