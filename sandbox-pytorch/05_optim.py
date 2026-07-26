## The goal here is to approximate the sin function by a polynomial function
## Variation from the pytorch program, but with a NN module and RMS optimization

import torch
dtype = torch.float 
device = torch.device('cpu')

#Random input/outputs data 
N = 2000
x = torch.linspace(-torch.pi, torch.pi, N, device=device, dtype=dtype)
y = torch.sin(x)

#We write the coeffcients of the polynomial in a NN layer
p = torch.tensor([1, 2, 3])

#And vectorize our array using unsqueeze
#x.unsqueeze(-1) = (2000,1)
#p seen as (1,3)
#Broadcasting to get (2000,3), so for each xi, we compute [xi, xi**2, xi**3]  
xx = x.unsqueeze(-1).pow(p)

#We define our model as a sequence of layers
#Linear layer to compute ouputs using a linear functon 
#Flatten layer to flaten the output of the linear layer to a 1D tensor
model = torch.nn.Sequential(
    torch.nn.Linear(3,1),
    torch.nn.Flatten(0,1)
)

#We define the  loss function using the MSE
loss_fn = torch.nn.MSELoss(reduction='sum')
learning_rate = 1e-3

#Define an Optimizer that will update the weights of the model for us
optimizer=torch.optim.RMSprop(model.parameters(), lr=learning_rate)

for t in range (N):
    #Forward pass: called like a function, tensor input -> tensor output
    y_pred = model(xx)

    #Compute and print loss
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable
    # weights of the model)    
    optimizer.zero_grad()

    #Backward pass: compute gradient of the loss with respect to the parameters
    #Same as autograd
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()

linear_layer=model[0]

#For a linear model, parameters as stored as weight and bias
print(f"Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3")
