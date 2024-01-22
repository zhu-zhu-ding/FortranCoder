import json
import re

import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import utils
def extract_fortran(fortran_code):
    pattern = r"```fortran(.*?)```"
    matches = re.findall(pattern, fortran_code, re.DOTALL)
    if matches:
        extracted_code = matches[0]
        return extracted_code
    else:
        return ''
function_list = utils.read_json('./alpaca_function/function_benchmark_alpaca.jsonl')
function_list = [data['code'] for data in function_list]
subroutine_list = utils.read_json('./alpaca_subroutine/subroutine_benchmark_alpaca.jsonl')
subroutine_list = [data['code'] for data in subroutine_list]
test_function = utils.read_json('')
test_function = [extract_fortran(data['code']) for data in test_function]

test_subroutine = utils.read_json('')
test_subroutine = [extract_fortran(data['code']) for data in test_subroutine]



rouge_l_list = []
rouge_1_list = []
rouge_2_list = []
fourgram_bleu_list=[]
rouge_1_score = 0
rouge_2_score = 0
rouge_l_score = 0
unigram_bleu = 0
bigram_bleu = 0
trigram_bleu = 0
fourgram_bleu = 0
smooth_func = SmoothingFunction().method7
for i,code in enumerate(function_list):
    bleu_score = nltk.translate.bleu_score.SmoothingFunction().method1
    fourgram_bleu += nltk.translate.bleu_score.sentence_bleu(function_list[i], test_function[i],
                                                             weights=(0, 0, 0, 1),
                                                             smoothing_function=bleu_score)
    # 计算ROUGE-1相似度
    rouge_1_score += sentence_bleu(function_list[i], test_function[i], weights=(1, 0, 0),
                                   smoothing_function=smooth_func)
    # 计算ROUGE-2相似度
    rouge_2_score += sentence_bleu(function_list[i], test_function[i], weights=(0.5, 0.5, 0),
                                   smoothing_function=smooth_func)
    # 计算ROUGE-L相似度
    rouge_l_score += sentence_bleu(function_list[i], test_function[i], weights=(0, 0, 1),
                                   smoothing_function=smooth_func)
for i,code in enumerate(subroutine_list):
    bleu_score = nltk.translate.bleu_score.SmoothingFunction().method1
    fourgram_bleu += nltk.translate.bleu_score.sentence_bleu(subroutine_list[i], test_subroutine[i],
                                                             weights=(0, 0, 0, 1),
                                                             smoothing_function=bleu_score)
    # 计算ROUGE-1相似度
    rouge_1_score += sentence_bleu(subroutine_list[i], test_subroutine[i], weights=(1, 0, 0),
                                   smoothing_function=smooth_func)
    # 计算ROUGE-2相似度
    rouge_2_score += sentence_bleu(subroutine_list[i], test_subroutine[i], weights=(0.5, 0.5, 0),
                                   smoothing_function=smooth_func)
    # 计算ROUGE-L相似度
    rouge_l_score += sentence_bleu(subroutine_list[i], test_subroutine[i], weights=(0, 0, 1),
                                   smoothing_function=smooth_func)
total_len = len(function_list)+subroutine_list
print(f"rouge1: {rouge_1_score/total_len},"
      f"rouge2: {rouge_2_score/total_len},"
      f"rouge1: {rouge_l_score/total_len},"
      f"bleu4: {fourgram_bleu/total_len}")