#%% packages
import torch
import seaborn as sns
import numpy as np


# %%
x= torch.tensor(5.5,requires_grad=True)  # calculate gardient for us automaatically
# %%
y = x + 10
print(y)

# %%
def y_function(val):
    return (val-3) * (val-6) * (val-4)
x_range =np.linspace(0,10,101)
y_range = [y_function(i) for i in x_range]
sns.lineplot(x=x_range,y=y_range)


# %%
y = (x-3)*(x-4)*(x-6)
y.backward()  # caculate gradient
print(x.grad)  # this is the gradient
# %%
# Each tensor is like a node in a layer
# first layer
x11 = torch.tensor(2.0, requires_grad=True)
x21 = torch.tensor(3.0, requires_grad=True)

# second hidden layer
x12 = 5 * x11 - 3 * x21
x22 = 2 * x11**2 + 2 * x21

# output layer
y = 4 * x12 + 3 * x22
y.backward()
print(x11.grad)
print(x21.grad)
# %%
