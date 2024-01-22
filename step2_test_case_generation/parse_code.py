import re
import json

import unit_test_utils
import utils
from tqdm import tqdm

read_function_path = '../step1_extract_highquility_code_segment/data/step1_function_result.jsonl'
read_subroutine_path = '../step1_extract_highquility_code_segment/data/step1_subroutine_result.jsonl'

save_function_path = 'data/parse_function_result.jsonl'
save_subroutine_path = 'data/parse_subroutine_result.jsonl'

def parse_format(parse):
    parse = parse.split(',')
    parse = [item for item in parse if 'intent' not in item]
    result = ''
    for i,item in enumerate(parse):
        if i==len(parse)-1:
            result+=item
        else:
            result+=item+','
    return result
def parse_param(param):
    flag = False
    result = []
    temp = ''
    for i in param:
        if i!=',':
            temp+=i
            if i =='(':
                flag = True
            elif i ==')':
                flag = False
        else:
            if flag:
                temp += i
            else:
                result.append(temp.strip())
                temp =''
    result.append(temp)
    return result
def parse_fortran_function(function_str):
    try:
        # 获取code_line_list
        line_list = [code_line.split('!')[0].strip() for code_line in function_str.split('\n') if code_line.strip() and code_line.split('!')[0].strip()]
        if re.search(r'\bresult\b', line_list[0]):
            # 定义正则表达式模式来匹配变量声明
            pattern_name = r"\((.*?)\)"
            name_list = re.findall(pattern_name, line_list[0])
            in_name_list = [in_name.strip() for in_name in name_list[0].split(',')]
            out_name_list = [in_name.strip() for in_name in name_list[1].split(',')]

            variable_list = {}
            type_in = []
            type_out = []
            for code_line in line_list:
                if '::' in code_line:
                    temp_list = [i.strip() for i in code_line.split('::')]
                    for in_name in in_name_list:
                        if in_name in [temp.strip() for temp in temp_list[1].split(',')]:
                            type_in.append(parse_format(temp_list[0]))
                    for out_name in out_name_list:
                        if out_name in [temp.strip() for temp in temp_list[1].split(',')]:
                            type_out.append(temp_list[0])
            if len(type_in)!=len(in_name_list):
                return None
            if type_in and type_out:
                variable_list['type_in_num'] = len(type_in)
                variable_list['type_in'] = type_in
                variable_list['type_out_num'] = len(type_out)
                variable_list['type_out'] = type_out
            return variable_list
        else:
            variable_list = {}
            function_name = unit_test_utils.get_function_name(function_str)
            pattern_name = r"\((.*?)\)"
            name_list = re.findall(pattern_name, line_list[0])
            in_name_list = [in_name.strip() for in_name in name_list[0].split(',')]
            type_in = []
            type_out= []
            for code_line in line_list:
                if '::' in code_line:
                    temp_list = [i.strip() for i in code_line.split('::')]
                    for in_name in in_name_list:
                        if in_name in [temp.strip() for temp in temp_list[1].split(',')]:
                            type_in.append(parse_format(temp_list[0]))
                    if function_name in [temp.strip() for temp in temp_list[1].split(',')]:
                            type_out.append(parse_format(temp_list[0]))
            if len(type_out):
                type_list = ['real','integer','complex','logical','character']
                for item in line_list[0].split(' '):
                    for type in type_list:
                        if type in item:
                            type_out.append(item)
                            break
            if len(type_in)!=len(in_name_list):
                return None
            if type_in and type_out:
                variable_list['type_in_num'] = len(type_in)
                variable_list['type_in'] = type_in
                variable_list['type_out_num'] = len(type_out)
                variable_list['type_out'] = type_out
            return variable_list
    except:
        return None
def parse_fortran_subroutine(function_str):
    try:
        # 获取code_line_list
        line_list = [code_line.split('!')[0].strip() for code_line in function_str.split('\n') if code_line.strip() and code_line.split('!')[0].strip()]
        # 定义正则表达式模式来匹配变量声明
        pattern_name = r"\((.*?)\)"
        param_list = re.findall(pattern_name, line_list[0])
        param_list = [param.strip() for param in param_list[0].split(',')]
        # in_name_list = [in_name.strip() for in_name in name_list[0].split(',')]
        # out_name_list = [in_name.strip() for in_name in name_list[1].split(',')]
        variable_list = {}
        type_in = []
        type_out = []
        param_name = []
        for code_line in line_list:
            if '::' in code_line:
                temp_list = {'type':code_line.split('::')[0].strip(),
                             'param':[item.strip() for item in parse_param(code_line.split('::')[1])]}
                if 'intent(in)' in temp_list['type']:
                    type_in.extend([parse_format(temp_list['type'])]*len(temp_list['param']))
                    param_name.extend(temp_list['param'])
                elif 'intent(out)' in temp_list['type']:
                    type_out.extend([parse_format(temp_list['type'])]*len(temp_list['param']))
                    param_name.extend(temp_list['param'])
        if type_in and type_out and (len(type_in)+len(type_out)==len(param_list)):
            variable_list['param_name'] = param_name
            variable_list['type_in_num'] = len(type_in)
            variable_list['type_in'] = type_in
            variable_list['type_out_num'] = len(type_out)
            variable_list['type_out'] = type_out
        return variable_list
    except:
        return None
def main():
    function_list = utils.read_json(read_path=read_function_path)
    function_list = [data['function'] for data in function_list]
    subroutine_list = utils.read_json(read_path=read_subroutine_path)
    subroutine_list = [data['subroutine'] for data in subroutine_list]

    subroutine_list = [{'subroutine':subroutine,'param_info':parse_fortran_subroutine(subroutine)} for i,subroutine in enumerate(subroutine_list)
                   if parse_fortran_subroutine(subroutine)]
    function_list = [{'function':function,'param_info':parse_fortran_function(function)} for i,function in enumerate(function_list)
                   if parse_fortran_function(function)]
    subroutine_list = [{'id':i,'param_info':subroutine['param_info'],'code':subroutine['subroutine']} for i,subroutine in enumerate(subroutine_list)]
    function_list = [{'id':i,'param_info':function['param_info'],'code':function['function']} for i,function in enumerate(function_list)]

    utils.save_json(save_path=save_subroutine_path,save_list=subroutine_list)
    utils.save_json(save_path=save_function_path,save_list=function_list)
if __name__ == "__main__":
    main()

