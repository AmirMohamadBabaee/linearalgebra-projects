#!/usr/bin/env python
# coding: utf-8

import numpy as np

# define function to calculate LU factor of matrix A

def calculate_LU(A):
    length = np.shape(A)[0]
    L = np.eye(length)
    pivot_col = 0
    pivot_row = 0
    L_col = 0
    while pivot_col < length and pivot_row < length:
        while A[pivot_row, pivot_col] == 0:
            pivot_col += 1
            if pivot_col > length:
                return
        for i in range(pivot_row + 1, length):
            L[i, pivot_col] = A[i, pivot_col] / A[pivot_row, pivot_col]
            A[i] = A[i] - L[i, pivot_col] * A[pivot_row]
        L_col += 1
        pivot_col += 1
        pivot_row += 1
    return L, A       # A is equal to U in this step


# solve Ly = b by forward substitute method

def solve_forward_substitute(L, b):
    y = np.zeros((np.size(b), 1))
    for i in range(np.shape(L)[0]):
        y[i] += b[i]
        for j in range(i):
            y[i] -= L[i, j] * y[j]
        y[i] /= L[i, i]
    return y

# solve Ux = y by backward substitute method

def solve_backward_substitute(U, y):
    x = np.zeros((np.size(y), 1))
    for i in range(np.shape(U)[0] - 1 , -1, -1):
        x[i] += y[i]
        for j in range(np.shape(U)[0] - 1, i, -1):
            x[i] -= U[i, j] * x[j]
        x[i] /= U[i, i]
    return x


def solve_matrix_equation(A, b):
    A_array = np.array(A)
    LU = calculate_LU(A_array)
    L = LU[0]
    U = LU[1]
    y = solve_forward_substitute(L, b)
    x = solve_backward_substitute(U, y)
    return x


n, m = list(map(int, input().split()))
A = list()
b_list = list()
x_list = list()
for _ in range(n):
    A.append(list(map(float, input().split())))
for i in range(m):
    b_list.append(list(map(float, input().split())))
    x_list.append(solve_matrix_equation(np.array(A), b_list[i]).reshape(1, n))
for x in x_list:
    print(*list(map(lambda a: round(a, 2), x.tolist()[0])), sep=' ')
