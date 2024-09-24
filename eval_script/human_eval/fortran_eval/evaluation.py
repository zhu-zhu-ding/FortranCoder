from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union, Iterable, Dict
import itertools

import numpy as np
import tqdm

from fortran_eval.data import FORTRAN_EVAL, read_problems, stream_jsonl, write_jsonl
from fortran_eval.execution import check_correctness


def estimate_pass_at_k(
        num_samples: Union[int, List[int], np.ndarray],
        num_correct: Union[List[int], np.ndarray],
        k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def evaluate_functional_correctness(
        sample_file: str,
        k: List[int] = [1, 10, 100],
        n_workers: int = 4,
        timeout: float = 3.0,
        problem_file: str = FORTRAN_EVAL,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """

    problems = read_problems(problem_file)

    # Check the generated samples against test suites.
    with ThreadPoolExecutor(max_workers=n_workers) as executor:

        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)

        print("Reading samples...")
        for sample in tqdm.tqdm(stream_jsonl(sample_file)):
            task_id = sample["task_id"]
            completion = sample["completion"]
            args = (problems[task_id], completion, timeout, completion_id[task_id])
            future = executor.submit(check_correctness, *args)
            futures.append(future)
            completion_id[task_id] += 1
            n_samples += 1

        assert len(completion_id) == len(problems), "Some problems are not attempted."

        print("Running test suites...")
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))

    # Calculate pass@k.
    total, correct, compile_sum, total_average_time = [], [], [], []
    for result in results.values():
        print(result)
        result.sort()
        exe_time = [r[1]["time"] for r in result]
        passed = [r[1]["passed"] for r in result]
        exe_res = [r[1]["result"] for r in result]
        compile_passed = 0
        average_time = 0.0
        for i in exe_res:
            if i != "compile failed":
                compile_passed += 1
        for i in range(len(exe_res)):
            if exe_res[i] == "passed":
                average_time += exe_time[i]
        if sum(passed) > 0:
            average_time = average_time / sum(passed)
        else:
            average_time = 0.0
        total_average_time.append(average_time)
        compile_sum.append(compile_passed)
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)
    compile_sum = np.array(compile_sum)
    total_average_time = np.array(total_average_time)
    # print(total_average_time)
    # print(total_average_time.mean())
    ks = k
    pass_at_k = {f"correct_pass@{k}": estimate_pass_at_k(total, correct, k).mean()
                 for k in ks if (total >= k).all()}

    compile_at_k = {f"compile_pass@{k}": estimate_pass_at_k(total, compile_sum, k).mean()
                    for k in ks if (total >= k).all()}

    pass_compile_at_k = {f"correct_compile_pass@{k}": estimate_pass_at_k(total, correct, k).mean() / estimate_pass_at_k(total, compile_sum, k).mean()
                         for k in ks if (total >= k).all()}
    # print(total,correct,compile_sum)
    # Finally, save the results in one file:
    def combine_results():
        for sample in stream_jsonl(sample_file):
            task_id = sample["task_id"]
            result = results[task_id].pop(0)
            sample["result"] = result[1]["result"]
            sample["passed"] = result[1]["passed"]
            sample["time"] = result[1]["time"]
            yield sample

    out_file = sample_file + "_results.jsonl"
    print(f"Writing results to {out_file}...")
    write_jsonl(out_file, tqdm.tqdm(combine_results(), total=n_samples))

    return pass_at_k, compile_at_k, pass_compile_at_k, total_average_time.mean()
