import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import linalg as LA
import math


#def gradient_descent(grad_func, func, start_point, steps, step_size):
#    to_return = []
#    to_return.append("Step\tSSE\tNorm")
#    temp = start_point
#    for i in range(0,steps):
#        to_return.append(steps,'\t')
#        grad_point = grad_func(temp)
#        new_temp = [None]*len(start_point)
#        for n in range(0,len(start_point)):
#            new_temp[n] = temp[n]-step_size*grad_point[n]
#        #new_temp = (temp[0]-step_size*grad_point[0],temp[1]-step_size*grad_point[1])
#        new_temp = tuple(new_temp)
#        to_return.append(func(new_temp))
#        temp = new_temp
#    return to_return

#def incremental_gradient_descent(grad_func, func, alpha_vec, steps, step_size):
#    alpha = alpha_vec
#    i = 1
#    while(np.linalg.norm(grad_func(alpha).norm) <= )

#def get_alpha(X_matrix_with_offset,y):
#    temp = np.dot(np.transpose(X_matrix_with_offset),X_matrix_with_offset)
#    temp = np.linalg.inv(temp)
#    temp = np.dot(temp,np.transpose(X_matrix_with_offset))
#    temp = np.dot(temp,y)
#    return temp

def hypothesis(theta, X, n):
    h = np.ones((X.shape[0],1))
    theta = theta.reshape(1,n)
    for i in range(0,X.shape[0]):
        h[i] = float(np.matmul(theta, X[i]))
    h = h.reshape(X.shape[0])
    return h

def BGD(theta, alpha, num_iters, h, X, y, n, to_return):
    cost = np.ones(num_iters)
    for i in range(0,num_iters):
        to_return.append(i+1)
        to_return.append(',')
        theta[0] = theta[0] - (alpha/X.shape[0]) * sum(h - y)
        for j in range(1,n):
            theta[j] = theta[j] - (alpha/X.shape[0]) * sum((h-y) * X.transpose()[j])
        #this_theta = theta
        h = hypothesis(theta, X, n)
        cost[i] = (1/X.shape[0]) * 0.5 * sum(np.square(h - y))
        to_return.append(cost[i])
        to_return.append(',')
        to_return.append(np.linalg.norm(theta))
        to_return.append(',')
        to_return.append(str(theta))
        to_return.append('\n')
    theta = theta.reshape(1,n)
    return theta, cost

def BGD_linear_regression(X, y, learning_rate, num_iters):
    to_return = ["Step,SSE,Norm,Alpha\n"]

    n = X.shape[1]
    #one_column = np.ones((X.shape[0],1))
    #X = np.concatenate((one_column, X), axis = 1)
    # initializing the parameter vector...
    theta = np.zeros(n)
    # hypothesis calculation....
    h = hypothesis(theta, X, n)
    # returning the optimized parameters by Gradient Descent...
    theta, cost = BGD(theta,learning_rate,num_iters,h,X,y,n, to_return)
    return theta, cost, to_return

def incremental_linear_regression(X, y, learning_rate, num_iters):
    to_return = ["Step,SSE,Norm,Alpha\n"]

    n = X.shape[1]

    theta = np.zeros(n)
    theta, cost = IGD(theta, learning_rate, num_iters, X, y, n,to_return)
    return theta, cost, to_return

def IGD(theta, alpha, num_iters, X, y, n, to_return):
    cost = np.ones(num_iters)
    data_size = X.shape[0]
    for i in range(0,num_iters):
        to_return.append(i+1)
        to_return.append(',')
        if(i >= data_size):
            i %= data_size
        preds = sigmoid(np.dot(X[i], theta))

        cost[i] = preds - y[i]

        gradient = np.dot(np.transpose(X[i]), cost[i]) / data_size

        theta -= alpha * gradient
        to_return.append(cost[i])
        to_return.append(',')
        to_return.append(np.linalg.norm(theta))
        to_return.append(',')
        to_return.append(str(theta))
        to_return.append('\n')

    return theta, cost

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def vec_print(data_list):
    print(r'\begin{bmatrix}\n')
    for i in range(0,len(data_list)):
        print(data_list[i])
        print(r'\\', '\n')
    print(r'\end{bmatrix}')

def table_print(data_list):
    print(r'\begin{tabular}{|c|c|}\n')
    print(r'\hline')
    for i in range(0,len(data_list)):
        print('{} & {}'.format(i, data_list[i]))
        print(r'\\', '\n',r'\hline')
    print(r'\end{tabular}')

def low_rank_k(u,s,vh,num):
# rank k approx

    u = u[:,:num]
    vh = vh[:num,:]
    s = s[:num]
    s = np.diag(s)
    my_low_rank = np.dot(np.dot(u,s),vh)
    return my_low_rank



cwd = os.getcwd()
#x_path=r'C:\Users\Hunter Schmidt\Documents\Homework\University of Utah\Fall 2020\CS 3190\HW 3\x.csv'
#y_path=r'C:\Users\Hunter Schmidt\Documents\Homework\University of Utah\Fall 2020\CS 3190\HW 3\y.csv'
x_path = os.path.join(cwd,r'Data\x4.csv')
y_path = os.path.join(cwd,r'Data\y4.csv')
a_path = os.path.join(cwd,r'Data\A.csv')

x_data = np.genfromtxt(x_path,delimiter=',')
y_data = np.genfromtxt(y_path)
a_data = np.genfromtxt(a_path,delimiter=',')
#alpha = get_alpha(x_data,y_data)
#func = lambda point : (alpha[0] + alpha[1] * point[0] + alpha[2] * point[1] + alpha[3] * point[2])
#grad_func = lambda point : (alpha[1], alpha[2], alpha[3])

#print(gradient_descent(grad_func,func,(0,0,0),100,0.01))
alpha, error, to_print = BGD_linear_regression(x_data,y_data,0.01,100)
to_print = [str(i) for i in to_print]
#print(''.join(to_print))

alpha_IGD, error_IGD, to_print_IGD = incremental_linear_regression(x_data,y_data,0.01,100)
to_print_IGD = [str(i) for i in to_print_IGD]
print(''.join(to_print_IGD))


U, s, Vt = LA.svd(a_data)
print("The second singular value is {}".format(s[2]))
rank_a = 0
e_vals = []
for i in s:
    if not np.isclose(i,0.0):
        rank_a+=1
        e_vals.append(i**2)
    else:
        e_vals.append(0.0)
print("The rank of A is {}".format(rank_a))
V = np.transpose(Vt)
print("The eigenvectors of A^TA are:\n")
print(V)

print("The Eigenvalues of A^TA are:\n")
print(e_vals)

#s_k = s
#for i in range(3,len(s_k)):
#    s_k[i]=0.0
A_k = low_rank_k(U,s,Vt,3)

print('End')