# -*- coding: utf-8 -*-
import json

outFile = './result_wizardCoder_0.2.jsonl'
resultFile = './sample_wizardCoder_0.2.jsonl'
result_fp = open(resultFile, 'w')
fp = open(outFile, 'r', encoding="utf8")

i = 0


def generate_one_completion_starcoder(res):
    # global i
    # line = lines[(int)(i)]
    # i += 1
    # print(line)
    # res = json.loads(line,strict=False)
    str = res['generated_text']
    ind = str.find("MODULE test_module")
    if ind == -1:
        ind = str.find("module test_module")
    ind2 = str.find("END MODULE test_module")
    if ind2 == -1:
        ind2 = str.find("end module test_module")
    # ind2 = str.find("END MODULE test_module")
    # print(ind)
    # if ind == -1:
    #     ind = 0
    str = str[ind:ind2+22]
    # print(res['answer'])
    return str


for line in fp:
    try:
        ret = json.loads(line)
        result = {}
        result['task_id'] = ret['task_id']
        ret_code = generate_one_completion_starcoder(ret)
        result['completion'] = ret_code
        # print(ret_code)
        result_fp.write(json.dumps(result, ensure_ascii=False) + "\n")
        # text = ret['text']
        # results = text.split('Q:')
        # content = results[1].split('\n\nA:')
        # for item in content:
        #     print(item)
    except:
        print(line)
