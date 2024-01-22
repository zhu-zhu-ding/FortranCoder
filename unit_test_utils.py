import json
import subprocess
import re
import threading

import numpy
from tqdm import tqdm
from itertools import chain
import gpt_api_base
import utils
import os

def get_function_name(fortran_code):
    match = re.search(r"function\s+(\w+)", fortran_code)
    if match:
        function_name = match.group(1)
        return function_name
    else:
        return ''
def get_subroutine_name(fortran_code):
    match = re.search(r"subroutine\s+(\w+)", fortran_code)
    if match:
        function_name = match.group(1)
        return function_name
    else:
        return ''
def extract_subroutine(fortran_code):
    fortran_code = fortran_code.lower()
    # 使用正则表达式匹配以"!"开头且包含"function"或"Function"的行
    pattern = re.compile(r'^\s*![^:\n]*\b(?:subroutine|Subroutine)\b.*$', re.MULTILINE)
    # 通过替换去除匹配到的行
    cleaned_code = re.sub(pattern, '', fortran_code)
    subroutines = []
    subroutine_code = ''
    subroutine_flag = 0
    for code_line in cleaned_code.split('\n'):
        if ('subroutine' in code_line) and ('end subroutine' not in code_line):
            subroutine_flag+=1
            subroutine_code+=code_line+'\n'
        if  ('end subroutine' in code_line):
            subroutine_flag -= 1
            subroutine_code += code_line + '\n'
            if subroutine_flag==0:
                subroutines.append(subroutine_code)
                subroutine_code = ''
        if ('subroutine' not in code_line) and ('end subroutine' not in code_line) and subroutine_flag!=0:
            subroutine_code+=code_line+'\n'
    return subroutines
def extract_function(fortran_code):
    fortran_code = fortran_code.lower()
    # 使用正则表达式匹配以"!"开头且包含"function"或"Function"的行
    pattern = re.compile(r'^\s*![^:\n]*\b(?:function|Function)\b.*$', re.MULTILINE)
    # 通过替换去除匹配到的行
    cleaned_code = re.sub(pattern, '', fortran_code)
    functions = []
    function_code = ''
    function_flag = 0
    for code_line in cleaned_code.split('\n'):
        if ('function' in code_line) and ('end function' not in code_line):
            function_flag+=1
            function_code+=code_line+'\n'
        if  ('end function' in code_line):
            function_flag -= 1
            function_code += code_line + '\n'
            if function_flag==0:
                functions.append(function_code)
                function_code = ''
        if ('function' not in code_line) and ('end function' not in code_line) and function_flag!=0:
            function_code+=code_line+'\n'
    return functions
def check_iscompile(code):
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
    executable = 'compile'
    code_file = f'compile.f90'
    with open(code_file, 'w',encoding="utf-8") as file:
        file.write(test_code)
    try:
        subprocess.check_output(['gfortran', code_file, '-o',executable], stderr=subprocess.STDOUT)
        return True
    except Exception as e:
        # print(e)
        return False
    finally:
        clear_files(executable)
def get_subroutine_result(test_code):
    # 将Fortran代码保存到文件中
    with open("subroutine.f90", "w", encoding='utf-8') as f:
        f.write(test_code)
    f.close()
    # 运行文件获取结果
    executable = "subroutine"
    try:
        # 编译 Fortran 代码
        subprocess.run(['gfortran', "subroutine.f90", '-o', executable], stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False)
        # 运行 Fortran 可执行文件并获取输出结果
        result = subprocess.run([f"./{executable}"], check=False, stdout=subprocess.PIPE, text=True, timeout=10)
        output = result.stdout
        output = [out for out in output.split('\n') if out!='' or out==None]
        return output
    except Exception:
        return None
    finally:
        clear_files(executable)
def get_function_result(test_code):
    # 将Fortran代码保存到文件中
    with open("function.f90", "w", encoding='utf-8') as f:
        f.write(test_code)
    f.close()
    # 运行文件获取结果
    executable = "function"
    try:
        # 编译 Fortran 代码
        subprocess.run(['gfortran', "function.f90", '-o', executable],  stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=False)

        # 运行 Fortran 可执行文件并获取输出结果
        result = subprocess.run([f"./{executable}"], check=False, stdout=subprocess.PIPE,stderr=subprocess.PIPE, text=True, timeout=10)
        output = result.stdout
        output = [out for out in output.split('\n') if out!='' or out==None]
        return output
    except Exception:
        return None
    finally:
        clear_files(executable)
def clear_files(executable):
    if os.path.exists(executable):
        os.remove(executable)
    if os.path.exists(f'{executable}.f90'):
        os.remove(f'{executable}.f90')
    if os.path.exists(f'{executable}.exe'):
        os.remove(f'{executable}.exe')
    for root, dirs, files in os.walk('./'):
        for file in files:
            if file.endswith(".mod") or file.endswith(".txt"):
                file_path = os.path.join(root, file)
                os.remove(file_path)
