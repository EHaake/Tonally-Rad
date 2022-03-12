'''
A preconditioned version of the Conjugate Gradients method in order to improve the convergence rate.
This implementation uses an Incomplete Cholesky factorization of the A matrix as the preconditioner.

Solve the system Ax = b using the preconditioned Conjugate Gradients method.

Source: http://www.grimnes.no/algorithms/preconditioned-conjugate-gradients-method-matrices/
'''

import numpy as np
import math

MAX_ITERATIONS = 10 ** 4
MAX_ERROR = 10 ** -3

A = np.array([[3, 1, 0, 0],
              [1, 4, 1, 3],
              [0, 1, 10, 0],
              [0, 3, 0, 3]], dtype=float)

x = np.array([[2, 3, 4, 5]], dtype=float).T


# b = np.array([[1, 1, 1, 1]], dtype=float).T
b = np.array([[0, 0, 1, 0]], dtype=float).T


def incomplete_cholesky_factorization(A):
    '''
    Compute the incomplete Cholesky factorization of A
    :param A: a positive definite matrix
    '''
    mat = np.copy(A)
    n = mat.shape[1]

    for k in range(n):
        mat[k, k] = math.sqrt(mat[k, k])
        for i in range(k+1, n):
            if mat[i, k] != 0:
                mat[i, k] = mat[i, k] / mat[k, k]
        for j in range(k+1, n):
            for i in range(j, n):
                if mat[i, j] != 0:
                    mat[i, j] = mat[i, j] - mat[i, k] * mat[j, k]
    for i in range(n):
        for j in range(i+1, n):
            mat[i, j] = 0

    return mat


def solve_conjugate_gradients(A, x, b):
    residual = b - A.dot(x)

    # compute the incomplete Cholesky factorization of A as a preconditioner
    preconditioner = np.linalg.inv(incomplete_cholesky_factorization(A))

    z = np.dot(preconditioner, residual)
    d = z

    error = np.dot(residual.T, residual)

    iteration = 0
    while iteration < MAX_ITERATIONS and error > MAX_ERROR ** 2:
        q = np.dot(A, d)
        a = np.dot(residual.T, z) / np.dot(d.T, q)

        phi = np.dot(z.T,  residual)
        old_res = residual

        x = x + a * d
        residual = residual - a * q

        z = np.dot(preconditioner, residual)
        beta = np.dot(z.T, (residual-old_res)) / phi  # Polak-Ribiere
        d = z + beta * d

        error = residual.T.dot(residual)

        iteration += 1

    if iteration < MAX_ITERATIONS:
        print('Precision achieved. Iterations:', iteration)
    else:
        print('Convergence failed.')

    return x


# print(np.dot(A, solve_conjugate_gradients(A, x, b)), "==", b)
