import os

import fire
import sys

sys.path.append('/home/data1/wpd1/Fortran/human_eval')
from fortran_eval.data import FORTRAN_EVAL
from fortran_eval.evaluation import evaluate_functional_correctness


def entry_point(
        sample_file: str = "",
        k: str = "1,10,100",
        n_workers: int = 4,
        timeout: float = 3.0,
        problem_file: str = FORTRAN_EVAL,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))
    results1, results2, results3, mean_time = evaluate_functional_correctness(sample_file, k, n_workers, timeout, problem_file)
    print(results1, results2, results3)
    print("passed_mean_time: ", mean_time)


def main():
    fire.Fire(entry_point)


sys.exit(main())
