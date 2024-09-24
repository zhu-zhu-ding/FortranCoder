import re
from copy import deepcopy
import random
import time
import json
from tqdm import tqdm
import gpt_api_base
import utils
import unit_test_utils
read_path = ''
save_path = ''
def set_unit_test_prompt(fortran_code):
    return f"""As a Fortran expert, you need to write 5 different test cases for the following Fortran function and output the results.Fortran fucntion code as follows:
        '''
        {fortran_code}
        '''
        You must comply with the following requirements:
        1.You only need to generate a test case but you have to make sure that the test case is correct , comprehensive and complex.
        2.You only need to use Fortran's output function to output the output of the function, no other additional information is needed.Each test result output occupies 1 line exclusively.
        3.You must follow the following template:
        '''
        program main
        use test_function
        !to complete the test code
        end program main
        '''
        """
def gen_unit_test_prompt(fortran_code):
    answer_ex = f"""
program main
  use test_function
  implicit none
  integer :: m
  real, allocatable :: arr(:)
  
  !case1
  m=2
  allocate(arr(m))
  arr = [-1.042, 2.421]
  print *, sum_array(m, arr)
  deallocate(arr)

  !case2
  m=4
  allocate(arr(m))
  arr = [-1.0, 2.0, -3.0, 4.0]
  print *, sum_array(m, arr)
  deallocate(arr)

  !case3
  m=5
  allocate(arr(m))
  arr = [100.0, 200.0, 300.0, 400.0, 500.0]
  print *, sum_array(m, arr)
  deallocate(arr)

  !case4
  m=3
  allocate(arr(m))
  arr = [0.0, 0.0, 0.0]
  print *, sum_array(m, arr)
  deallocate(arr)

  !case5
  m=5
  allocate(arr(m))
  arr = [2.3132, -1.32, 2.5354, 0.3192, -1.23912]
  print *, sum_array(m, arr)
  deallocate(arr)
end program main
    """
    code_ex = """      
    function sum_array(m,arr) result(total)
      implicit none
      integer,intent(in):: m
      real, dimension(m), intent(in) :: arr
      real :: total
      integer :: i

      total = 0.0
      do i = 1, size(m)
        total = total + arr(i)
      end do
    end function sum_array
                        """
    message = [
        # {"role": "system", "content": code_ex},
        {"role": "user", "content": set_unit_test_prompt(code_ex)},
        {"role": "assistant", "content": answer_ex},
        {"role": "user", "content": set_unit_test_prompt(fortran_code)}
    ]
    return message
def extract_fortran(fortran_code):
    pattern = r"```fortran(.*?)```"
    matches = re.findall(pattern, fortran_code, re.DOTALL)
    if matches:
        extracted_code = matches[0]
        return extracted_code
    else:
        return fortran_code
def result_filter(result):
    ban_list = ['NaN','Infinity','-Infinity']
    for data in result:
        for item in data['result']:
            if item in ban_list:
                return False
    return True
def generate_test_case(data):
    n=3
    code = data['code']
    result = {}
    while(n>0):
        n-=1
        test_code_list = gpt_api_base.call_openai(message=gen_unit_test_prompt(code),temperature=0.8,n=1)
        print(test_code_list)
        for test_code in test_code_list:
            test_code = extract_fortran(test_code)
            temp_result = unit_test_utils.get_function_result(set_test_code(code,test_code))
            if temp_result ==None and result_filter(temp_result)==False:
                result = {'unit_test': test_code, 'result': None}
                continue
            else:
                result={'unit_test':test_code,'result':temp_result}
                break
    return result
def set_test_code(function,test_code):
    code = f"""
    module test_function
    contains
    {function}
    end module test_function
    {test_code}
    """
    return code
def main():
    data_list = utils.read_json(read_path=read_path)
    data_list = data_list[:5]
    result_list = [{'id':i,
                    'param_info':data['param_info'],
                    'test_case':generate_test_case(data=data),
                    'code':data['code']}
                   for i,data in tqdm(enumerate(data_list),total=len(data_list),desc='call gpt processing')
                   ]
    result_list = [data for data in result_list if data['test_case']['result']!=None]
    result_list = [{'id':i,
                    'param_info':data['param_info'],
                    'test_case':data['test_case'],
                    'code':data['code']} for i,data in enumerate(result_list)]

    utils.save_json(save_path = save_path,save_list=result_list)
if __name__ == "__main__":
    main()
