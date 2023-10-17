from typing import List, Callable

import time
import linalg
import numpy as np
import torch


def py_matrix_multiply(mat1: List[List[int]], mat2: List[List[int]]) -> List[List[int]]:
    if len(mat1[0]) != len(mat2):
        raise ValueError("Uncompatible matrix shapes")
    result = []
    for i in range(len(mat1)):
        row_result = []
        for j in range(len(mat2[0])):
            sum_val = 0
            for k in range(len(mat2)):
                sum_val += mat1[i][k] * mat2[k][j]
            row_result.append(sum_val)
        result.append(row_result)
    return result



def test_timings(func: Callable, *args):
    _ = func(*args)
    start_time = time.time()
    _ = func(*args)
    end_time = time.time()
    return round(end_time - start_time, 5)


def compare(matrix_size: int) -> None:
    matrix_a = np.random.rand(matrix_size, matrix_size)
    matrix_b = np.random.rand(matrix_size, matrix_size)

    list_a = matrix_a.tolist()
    list_b = matrix_b.tolist()

    print(
        "Mat mul (Pure Python), size={0}x{0}: {1} seconds".format(
            matrix_size, test_timings(py_matrix_multiply, list_a, list_b)
        )
    )
    print(
        "Mat mul (Pure C++), size={0}x{0}: {1} seconds".format(
            matrix_size, test_timings(linalg.LinearAlgebra.matmulPure, list_a, list_b)
        )
    )
    print(
        "Mat mul (C++ BLAS), size={0}x{0}: {1} seconds".format(
            matrix_size, test_timings(linalg.LinearAlgebra.matmulBlas, list_a, list_b)
        )
    )
    print(
        "Mat mul (Python numpy), size={0}x{0}: {1} seconds\n".format(
            matrix_size, test_timings(np.dot, np.array(list_a), np.array(list_b))
        )
    )

def compare_cosine(N: int, D: int) -> None:
    matrix_a = torch.randn(N, D)
    matrix_b = torch.randn(N, D)

    list_a = matrix_a.tolist()
    list_b = matrix_b.tolist()

    assert np.allclose(linalg.LinearAlgebra.cosineBlas(np.array(list_a),np.array(list_b),1e-8),
                       torch.nn.functional.cosine_similarity(matrix_a,matrix_b),1e-4)

    print(
        "Cosine similarity (C++ BLAS), size={}x{}: {} seconds\n".format(
            N,D,test_timings(linalg.LinearAlgebra.cosineBlas,np.array(list_a), np.array(list_b),1e-8)
        )
    )
    print(
        "Cosine similarity (torch), size={}x{}: {} seconds\n".format(
            N,D,test_timings(torch.nn.functional.cosine_similarity,matrix_a, matrix_b)
        )
    )



if __name__ == "__main__":
    for size in [10,50,100,300,500,700,5000]:
        compare_cosine(size,2*size)
