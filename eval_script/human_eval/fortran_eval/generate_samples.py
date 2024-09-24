import json
import os
import sys
sys.path.append(os.path.abspath('..'))
from fortran_eval.data import write_jsonl, read_problems


problems = read_problems()
# 每个prompt生成代码数
num_samples_per_task = 1


lines = []
i = 0

def generate_one_completion_v4(problem):
    global i
    line = lines[(int)(i)]
    i += 1
    # print(line)
    res = json.loads(line)
    # print(res['answer'])
    return res['answer']


def generate_one_completion_chatgpt(problem):
    global i
    line = lines[(int)(i)]
    i += 1
    # print(line)
    res = json.loads(line)
    print(res['response']['choices'][0]['message']['content'])
    str = res['response']['choices'][0]['message']['content']
    str = str.replace("```", "")
    ind = str.find("MODULE test_module")
    print(ind)
    str = str[ind:]
    return str


def generate_one_completion_chatglm(problem):
    global i
    line = lines[(int)(i)]
    i += 1
    # print(line)
    res = json.loads(line)
    str = res['answer']
    ind = str.find("MODULE test_module")
    ind2 = str.find("END MODULE test_module")
    print(ind)
    if ind == -1:
        ind = 0
    str = str[ind:ind2+22]
    # print(res['answer'])
    return str


def generate_one_completion_starcoder(problem):
    global i
    line = lines[(int)(i)]
    i += 1
    print(line)
    res = json.loads(line,strict=False)
    str = res['generated_text']
    ind = str.find("MODULE test_module")
    if ind == -1:
        ind = str.find("module test_module")
    # ind2 = str.find("END MODULE test_module")
    # print(ind)
    # if ind == -1:
    #     ind = 0
    str = str[ind:]
    # print(res['answer'])
    return str

def generate_one_completion_codegen7b(problem):
    global i
    line = lines[(int)(i)]
    i += 1
    print(line)
    res = json.loads(line,strict=False)
    str = res['answer']
    ind2 = str.find("END MODULE test_module")
    str = str[ind2 + 22:]
    ind = str.find("MODULE test_module")
    if ind == -1:
        ind = str.find("module test_module")
    ind2 = str.find("END MODULE test_module")
    if ind2 == -1:
        ind2 = str.find("end module test_module")
    # print(ind)
    if ind == -1:
        ind = 0
    str = str[ind:ind2+22]

    # print(res['answer'])
    return str

def generate_one_completion_wizardcoder(problem):
    global i
    line = lines[(int)(i)]
    i += 1
    # print(line)
    res = json.loads(line,strict=False)
    str = res['answer']
    ind2 = str.find("END MODULE test_module")
    if ind2 == -1:
        ind2 = str.find("end module test_module")
    str = str[ind2 + 22:]
    if str == "":
        print(i)
    ind = str.find("MODULE test_module")
    if ind == -1:
        ind = str.find("module test_module")
    ind2 = str.find("END MODULE test_module")
    if ind2 == -1:
        ind2 = str.find("end module test_module")
    # print(ind)
    if ind == -1:
        ind = 0
    str = str[ind:ind2+22]

    # print(res['answer'])
    return str

def generate_one_completion_codert5(problem):
    global i
    line = lines[(int)(i)]
    i += 1
    print(line)
    res = json.loads(line,strict=False)
    str = res['answer']

    ind2 = str.find("END MODULE test_module")
    str = str[ind2 + 22:]
    ind = str.find("MODULE test_module")
    if ind == -1:
        ind = str.find("module test_module")
    ind2 = str.find("END MODULE test_module")
    if ind2 == -1:
        ind2 = str.find("end module test_module")
    # print(ind)
    if ind == -1:
        ind = 0
    str = str[ind:ind2+22]

    # print(res['answer'])
    return str

def generate_one_completion(problem):

    return problem['canonical_solution']

# with open("../FortranEval/5-1-30.jsonl", "r",encoding='utf-8') as fp:
#     lines = fp.readlines()

samples = [
    dict(task_id=task_id, completion=generate_one_completion_starcoder(problems[task_id]))
    for task_id in problems
    for _ in range(num_samples_per_task)
]
write_jsonl("samples.jsonl", samples)
print("generate finished!")