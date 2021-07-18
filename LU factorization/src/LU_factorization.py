#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np


# In[25]:


# define function to calculate LU factor of matrix A

def calculate_LU(A):
    length = np.shape(A)[0]
    L = np.eye(length)
    
    pivot_row ,pivot_col = 0, 0
    L_col = 0
    while pivot_col < length and pivot_row < length:
        while A[pivot_row, pivot_col] == 0:
            pivot_col += 1
            if pivot_col > length:
                return
        for i in range(pivot_row + 1, length):
            L[i, pivot_col] = A[i, pivot_col] / A[pivot_row, pivot_col]
            A[i] = A[i] - L[i, pivot_col]*A[pivot_row]
        L_col += 1
        pivot_col += 1
        pivot_row += 1
    return L, A       # A is equal to U in this step


# # forward Substitution
# <span>
#     <img src="./images/Forward_Substitution_1.png">
#     <img src="./images/Forward_Substitution_2.png">
# </span>

# In[26]:


# solve Ly = b by forward substitute method

def solve_forward_substitute(L, b):
    y = np.zeros((np.size(b), 1))
    for i in range(np.shape(L)[0]):
        y[i] += b[i]
        for j in range(i):
            y[i] -= L[i, j] * y[j]
        y[i] /= L[i, i]
    return y


# # Backward Substitution
# <img src="./images/Backward_Substitution_1.png">
# <img src="./images/Backward_Substitution_2.png">

# In[27]:


# solve Ux = y by backward substitute method

def solve_backward_substitute(U, y):
    x = np.zeros((np.size(y), 1))
    for i in range(np.shape(U)[0] - 1 , -1, -1):
        x[i] += y[i]
        for j in range(np.shape(U)[0] - 1, i, -1):
            x[i] -= U[i, j] * x[j]
        x[i] /= U[i, i]
    return x


# In[34]:


def solve_martix_equation(A, b):
    A_array = np.array(A)
    L, U = calculate_LU(A_array)
    y = solve_forward_substitute(L, b)
    x = solve_backward_substitute(U, y)
    return x
    


# In[29]:


res1 = calculate_LU(np.array([
    [1., 1., 1.],
    [2., 4., 4.], 
    [3., 7., 10.]
]))
res2 = calculate_LU(np.array([
    [5., 6., 2.],
    [4., 5., 2.],
    [2., 4., 8.]
]))

print('L =\n', res1[0], '\nU =\n', res1[1])
print('L =\n', res2[0], '\nU =\n', res2[1])


# In[38]:


n, m = list(map(int, input().split()))
A = list()
b_list = list()
x_list = list()
for _ in range(n):
    A.append(list(map(float, input().split())))
for i in range(m):
    b_list.append(list(map(float, input().split())))
    x_list.append(solve_martix_equation(np.array(A), b_list[i]).reshape(1, n))
for x in x_list:
    print(x)


# In[ ]:




