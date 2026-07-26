## The goal here is to approximate the sin function by a polynomial function
## Varaiation from the numpy/torch program, using autograd

import torch
dtype = torch.float 
device = torch.device('cpu')

#Random input/outputs data 
N = 2000
x = torch.linspace(-torch.pi, torch.pi, N, device=device, dtype=dtype)
y = torch.sin(x)

#Randomly initiate weights
a = torch.randn((),device=device, dtype=dtype, requires_grad=True)
b = torch.randn((),device=device, dtype=dtype, requires_grad=True)
c = torch.randn((),device=device, dtype=dtype, requires_grad=True)
d = torch.randn((),device=device, dtype=dtype, requires_grad=True)

initial_loss =1.
learning_rate = 1e-6

for t in range (N):
    #Forward pass: compute predicted y 
    y_pred = a + b*x + c*(x**2) + d*(x**3)

    #Compute and print loss
    loss = torch.square(y_pred - y).sum()

    #Initial loss
    if t==0:
        initial_loss = loss.item()

    if t % 100 == 99:
        print(f'Iteration t = {t:4d}  loss(t)/loss(0) = {round(loss.item()/initial_loss, 6):10.6f}  a = {a.item():10.6f}  b = {b.item():10.6f}  c = {c.item():10.6f}  d = {d.item():10.6f}')

    #Use autograd to compute the loss 
    loss.backward()

    with torch.no_grad():
        #Update weights with autograd
        a -= learning_rate*a.grad
        b -= learning_rate*b.grad
        c -= learning_rate*c.grad
        d -= learning_rate*d.grad

        #Zero the gradients after updating weights
        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

print(f"Result: y = {a.item()} + {b.item()} x + {c.item()}x^2 + {d.item()}x^3")
