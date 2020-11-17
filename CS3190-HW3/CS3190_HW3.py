import numpy as np
import matplotlib.pyplot as plt
import os

def simple_regression(x_pts, y_pts):
    x_avg = np.average(x_pts)
    y_avg = np.average(y_pts)
    x_vec = [x-x_avg for x in x_pts]
    y_vec = [y-y_avg for y in y_pts]

    a = np.dot(x_vec, y_vec)/np.dot(x_vec,x_vec)
    b = y_avg - a*x_avg

    return (a, b)

def simple_prediction(a,b,x):
    return a*x + b

def print_model(a,b):
    if(b<0):
        print('y={}x{}'.format(a,b))
    else:
        print('y={}x+{}'.format(a,b))

def expand_matrix(vector, p):
    n = vector.size
    to_return = np.empty((n,p))
    for row in range(0,n):
        for power in range(0,p):
            to_return[row,power] = vector[row] ** power
    return to_return

def poly_fit(x_data,y_data, deg):
    x_exp = np.matrix(expand_matrix(x_data,deg))

    y = np.matrix(y_data)
    temp = ((x_exp.T * x_exp).I * x_exp.T)

    coef = [0,0,0]
    for i in range(0,x_data.size):
        coef[0] += temp[0,i]*y[0,i]
        coef[1] += temp[1,i]*y[0,i]
        coef[2] += temp[2,i]*y[0,i]

    
    return tuple(coef)

def poly_prediction(coef, x):
    y=0
    for p in range(0,len(coef)):
        y+=coef[p]*(x**p)

    return y

def gradient_descent(grad_func, func, start_point, steps, step_size):
    to_return = []
    to_return.append(func(start_point))
    temp = start_point
    for i in range(0,steps):
        grad_point = grad_func(temp)
        new_temp = (temp[0]-step_size*grad_point[0],temp[1]-step_size*grad_point[1])
        to_return.append(func(new_temp))
        temp = new_temp
    return to_return

def f_1(point):
    x = point[0]
    y = point[1]

    val = (x - y) ** 2 + (x * y)

    return val

def f_2(point):
    x = point[0]
    y = point[1]

    val = (1 - (y-4))**2 + 35 * ((x+6)-((y-4)**2))**2

    return val

def f_1_grad(point):
    x = point[0]
    y = point[1]

    new_x = 2*x-y
    new_y = 2*y-x

    return(new_x,new_y)

def f_2_grad(point):
    x = point[0]
    y = point[1]

    new_x = 70 * (8*y+x-10-(y**2))
    new_y = 140*(y**3) -1680 *(y**2) + 5878 * y - 140 * y * x + 560 * x - 5592

    return(new_x,new_y)

def table_print(data_list):
    print(r'\begin{tabular}{|c|c|}\n')
    print(r'\hline')
    for i in range(0,len(data_list)):
        print('{} & {}'.format(i, data_list[i]))
        print(r'\\', '\n',r'\hline')
    print(r'\end{tabular}')

def vec_print(data_list):
    print(r'\begin{bmatrix}\n')
    for i in range(0,len(data_list)):
        print(data_list[i])
        print(r'\\', '\n')
    print(r'\end{bmatrix}')

        


NUM_POINTS = 100
cwd = os.getcwd()
#x_path=r'C:\Users\Hunter Schmidt\Documents\Homework\University of Utah\Fall 2020\CS 3190\HW 3\x.csv'
#y_path=r'C:\Users\Hunter Schmidt\Documents\Homework\University of Utah\Fall 2020\CS 3190\HW 3\y.csv'
x_path = os.path.join(cwd,r'Data\x.csv')
y_path = os.path.join(cwd,r'Data\y.csv')

x_data = np.genfromtxt(x_path)
y_data = np.genfromtxt(y_path)

a,b = simple_regression(x_data, y_data)
print('Simple Model:\n')
print_model(a,b)
print('\n Predictions:')
print('x=4, y={}'.format(simple_prediction(a,b,4)), '\n')
print('x=8.5, y={}'.format(simple_prediction(a,b,8.5)), '\n')

x_train = x_data[:80]
y_train=(y_data[:80])
train_a,train_b = simple_regression(x_train,y_train)
print('Simple Model with Training:\n')
print_model(train_a,train_b)
print('\n Predictions:')
print('x=4, y={}'.format(simple_prediction(train_a,train_b,4)), '\n')
print('x=8.5, y={}'.format(simple_prediction(train_a,train_b,8.5)), '\n')

full_residual_vec = []
training_residual_vec = []

full_training_vec = []
training_training_vec = []
for i in range(0,100):
    x = x_data[i]
    y=y_data[i]
    if(i>=80):
        full_residual_vec.append(y - simple_prediction(a,b,x))
        training_residual_vec.append(y - simple_prediction(train_a,train_b,x))
    else:
        full_training_vec.append(y - simple_prediction(a,b,x))
        training_training_vec.append(y - simple_prediction(train_a,train_b,x))

print('Residual Vector for full data model:\n')
vec_print(full_residual_vec)
print('Magnitude: ',np.linalg.norm(full_residual_vec))

print('Residual Vector for training data model:\n')
vec_print(training_residual_vec)
print('Magnitude: ',np.linalg.norm(training_residual_vec))

print('Residual Vector 2 norm for full model on training data:')
print('Magnitude: ',np.linalg.norm(full_training_vec))

print('Residual Vector 2 norm for training model on training data:')
print('Magnitude: ',np.linalg.norm(training_training_vec))
#1 (d)
new_matrix = expand_matrix(x_data, 3)
table_print(new_matrix[[0,1,2], :])

coefficients = poly_fit(x_train,y_train,3)
print('Polynomial coefficients:')
print(coefficients)

residuals = []
for i in range(0,100):
    x = x_data[i]
    y = y_data[i]
    residuals.append(y - poly_prediction(coefficients, x))

test_resid_norm = np.linalg.norm(residuals[80:])

train_resid_norm = np.linalg.norm(residuals[:80])

print('Residual norm for Test data:')
print(test_resid_norm)
print('Residual norm for training data:')
print(train_resid_norm)

three_a = gradient_descent(f_1_grad, f_1,(2,3),20,0.05)

three_b = gradient_descent(f_2_grad, f_2,(2,3),100,0.0015)

print('Gradient Descent values for f1 with 20 steps and gamma=.05')
table_print(three_a)

print('Gradient Descent values for f2 with 100 steps and gamma=.0015')
table_print(three_b)

print('Smallest value gradient descent for f1 with 20 steps and gamma=0.5')
test_f1_descent = gradient_descent(f_1_grad, f_1,(2,3),20,0.5)
table_print(test_f1_descent)

print('Smallest value gradient descent for f2 with 100 steps and gamma=0.0017')
test_f2_descent = gradient_descent(f_2_grad, f_2,(2,3),100,0.0017)
table_print(test_f2_descent)