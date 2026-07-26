## The goal here is to approximate the sin function by a polynomial function
## Coded by hand but with pytorch tensors

import torch
dtype = torch.float 
device = torch.device('cpu')

#Random input/outputs data 
N = 2000
x = torch.linspace(-torch.pi, torch.pi, N, device=device, dtype=dtype)
y = torch.sin(x)

#Randomly initiate weights
a = torch.randn((),device=device, dtype=dtype)
b = torch.randn((),device=device, dtype=dtype)
c = torch.randn((),device=device, dtype=dtype)
d = torch.randn((),device=device, dtype=dtype)

learning_rate = 1e-6

for t in range (N):
    #Forward pass: compute predicted y 
    y_pred = a + b*x + c*(x**2) + d*(x**3)

    #Compute and print loss
    loss = torch.square(y_pred - y).sum()
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

print(f"Result: y = {a.item()} + {b.item()} x + {c.item()}x^2 + {d.item()}x^3")
