## The goal here is to approximate the sin function by a polynomial function
## Coded by hand in numpy, gradients also

import numpy as np 

#Random input/outputs data 
N = 2000
x = np.linspace(-np.pi, np.pi, N)
y = np.sin(x)

#Randomly initiate weights
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

learning_rate = 1e-6

for t in range (N):
    #Forward pass: compute predicted y 
    y_pred = a + b*x + c*(x**2) + d*(x**3)

    #Compute and print loss
    loss = np.square(y_pred - y).sum()
    if t % 100 == 99:
        print(t, loss)

    #Backprop to compute gradients pf a,b,c,d with respect to loss
    grad_y_pred = 2.0*(y_pred-y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x**2).sum()
    grad_d = (grad_y_pred * x**3).sum()

    #Update weights
    a -= learning_rate*grad_a
    b -= learning_rate*grad_b
    c -= learning_rate*grad_c
    d -= learning_rate*grad_d

print(f"Result: y = {a} + {b} x + {c}x^2 + {d}x^3")

