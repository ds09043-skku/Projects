import numpy as np
import torch

# Seed (Actually, it's not needed since there is no randomness in the code)
np.random.seed(2021313549)
torch.manual_seed(2021313549)

# input
x1 = np.array([1.0, 2.0, 3.0])
x2 = np.array([4.0, 5.0, 6.0])

# weight
w1 = np.array([[0.1, 0.2, 0.3, 0.4], 
               [0.5, 0.6, 0.7, 0.8], 
               [0.9, 1.0, 1.1, 1.2]])

w2 = np.array([[0.2, 0.1], 
               [0.4, 0.5], 
               [0.6, 0.2], 
               [0.8, 0.7]])

# target
y1_target = np.array([0, 1])
y2_target = np.array([1, 0])

# numpy to torch
x1_tensor = torch.from_numpy(x1)
x2_tensor = torch.from_numpy(x2)
w1_tensor = torch.from_numpy(w1)
w1_tensor.requires_grad_(True)    # for tracking gradients
w2_tensor = torch.from_numpy(w2)
w2_tensor.requires_grad_(True)
y1_target_tensor = torch.from_numpy(y1_target)
y2_target_tensor = torch.from_numpy(y2_target)

# <Task 1>
print("\nTask 1: Implementing Neural Networks")
print("-" * 50)

# forward function for torch
def torch_forward(x, w1, w2):
    # input -> hidden
    net1 = torch.matmul(x, w1)    # Net input of ReLU
    h = torch.relu(net1)    # Output of ReLU
    
    # hidden -> output
    net2 = torch.matmul(h, w2)    # Net input of softmax
    o = torch.softmax(net2, dim=0)    # Output of softmax
    
    return o, h, net1, net2

# forward function for numpy
def numpy_forward(x, w1, w2):
    # input -> hidden
    net1 = np.dot(x, w1)
    h = np.maximum(0, net1)    # ReLU
    
    # hidden -> output
    net2 = np.dot(h, w2)
    o = np.exp(net2) / np.sum(np.exp(net2))    # softmax
    
    return o, h, net1, net2

# forward for torch
y1_tensor, h_tensor_x1, net1_tensor_x1, net2_tensor_x1 = torch_forward(x1_tensor, w1_tensor, w2_tensor)
y2_tensor, h_tensor_x2, net1_tensor_x2, net2_tensor_x2 = torch_forward(x2_tensor, w1_tensor, w2_tensor)

# forward for numpy
y1_numpy, h_numpy_x1, net1_numpy_x1, net2_numpy_x1 = numpy_forward(x1, w1, w2)
y2_numpy, h_numpy_x2, net1_numpy_x2, net2_numpy_x2 = numpy_forward(x2, w1, w2)

print("\n구현 결과:")
print(f"입력 x1 = {x1_tensor.numpy()}의 출력: {y1_tensor.detach().numpy()}")    # detach() for no gradient
print(f"입력 x2 = {x2_tensor.numpy()}의 출력: {y2_tensor.detach().numpy()}")

# <Task 2>
print("\nTask 2: 손실 함수와 그래디언트 계산")
print("-" * 50)

# cross entropy loss for torch
def cross_entropy_loss_torch(y_pred, y_true):
    return -torch.sum(y_true * torch.log(y_pred))

# cross entropy loss for numpy
def cross_entropy_loss_numpy(y_pred, y_true):
    return -np.sum(y_true * np.log(y_pred))

# backward for numpy
def backward_numpy(x, y_pred, y_true, h, w2):
    do = y_pred - y_true    # output layer error
    
    grad_w2 = np.outer(h, do)    # w2 gradient
    
    dh = np.dot(do, w2.T)    # hidden layer error
    dh *= (h > 0)    # ReLU derivative: h'(x) = 0 if x <= 0, 1 if x > 0
    
    grad_w1 = np.outer(x, dh)    # w1 gradient
    
    return grad_w1, grad_w2

# x1 y1 torch
loss_x1_tensor = cross_entropy_loss_torch(y1_tensor, y1_target_tensor)
loss_x1_tensor.backward(retain_graph=True)
dw1_x1_tensor = w1_tensor.grad.clone()    # clone() : avoid messing up original tensor (in-place operation)
dw2_x1_tensor = w2_tensor.grad.clone()
w1_tensor.grad.zero_()    # zero_() : reset gradient
w2_tensor.grad.zero_()

# x1 y1 numpy
loss_x1_numpy = cross_entropy_loss_numpy(y1_numpy, y1_target)
dw1_x1_numpy, dw2_x1_numpy = backward_numpy(x1, y1_numpy, y1_target, h_numpy_x1, w2)

# x2 y2 torch
loss_x2_tensor = cross_entropy_loss_torch(y2_tensor, y2_target_tensor)
loss_x2_tensor.backward(retain_graph=True)
dw1_x2_tensor = w1_tensor.grad.clone()
dw2_x2_tensor = w2_tensor.grad.clone()
w1_tensor.grad.zero_()
w2_tensor.grad.zero_()

# x2 y2 numpy
loss_x2_numpy = cross_entropy_loss_numpy(y2_numpy, y2_target)
dw1_x2_numpy, dw2_x2_numpy = backward_numpy(x2, y2_numpy, y2_target, h_numpy_x2, w2)

# result
print("\nPyTorch 구현 결과:")
print(f"입력 x1에 대한 손실: {loss_x1_tensor.item():.6f}")    # item() : convert tensor to python float
print(f"입력 x1에 대한 w1 그래디언트:\n{dw1_x1_tensor.detach().numpy()}")
print(f"입력 x2에 대한 손실: {loss_x2_tensor.item():.6f}")
print(f"입력 x2에 대한 w1 그래디언트:\n{dw1_x2_tensor.detach().numpy()}")

print("\nNumpy 구현 결과:")
print(f"입력 x1에 대한 손실: {loss_x1_numpy:.6f}")
print(f"입력 x1에 대한 w1 그래디언트:\n{dw1_x1_numpy}")
print(f"입력 x2에 대한 손실: {loss_x2_numpy:.6f}")
print(f"입력 x2에 대한 w1 그래디언트:\n{dw1_x2_numpy}")

# <Task 3>
print("\nTask 3: 100회 반복 학습 및 가중치 업데이트")
print("-" * 50)

# hyper parameter
learning_rate = 0.01
num_epochs = 100

# training
def train_torch():
    # initialize weights
    w1_train = w1_tensor.clone().detach().requires_grad_(True)
    w2_train = w2_tensor.clone().detach().requires_grad_(True)
    # clone() : clone the tensor
    # detach() : seperate new(cloned) tensor from the original tensor
    # requires_grad_(True) : enable gradient tracking again (for cloned tensor)
    # reason for all these: shallow copy of original tensor
    
    losses = []    # store total loss for each epoch
    
    for epoch in range(num_epochs):
        # forward with x1
        y1_pred, _, _, _ = torch_forward(x1_tensor, w1_train, w2_train)    # _ : not used (only want y1_pred)
        loss1 = cross_entropy_loss_torch(y1_pred, y1_target_tensor)
        
        y2_pred, _, _, _ = torch_forward(x2_tensor, w1_train, w2_train)
        loss2 = cross_entropy_loss_torch(y2_pred, y2_target_tensor)
        
        # total loss
        total_loss = loss1 + loss2
        losses.append(total_loss.item())
        
        # backward
        total_loss.backward()
        
        # update weights
        with torch.no_grad():    # no_grad() : pause gradient tracking inside the block
            # without no_grad(), next backward() will also compute this operation
            w1_train -= learning_rate * w1_train.grad
            w2_train -= learning_rate * w2_train.grad
            
        # initialize gradients, without this, backward() will compute gradients of previous epoch
        w1_train.grad.zero_()
        w2_train.grad.zero_()
    
    return w1_train.detach().numpy(), w2_train.detach().numpy(), losses

def train_numpy():
    w1_train = w1.copy()    # copy() : shallow copy for numpy
    w2_train = w2.copy()
    losses = []
    
    for epoch in range(num_epochs):
        # forward with x1
        y1_pred, h_x1, _, _ = numpy_forward(x1, w1_train, w2_train)
        loss1 = cross_entropy_loss_numpy(y1_pred, y1_target)
        
        y2_pred, h_x2, _, _ = numpy_forward(x2, w1_train, w2_train)
        loss2 = cross_entropy_loss_numpy(y2_pred, y2_target)
        
        # total loss
        total_loss = loss1 + loss2
        losses.append(total_loss)
        
        # backward
        dw1_x1, dw2_x1 = backward_numpy(x1, y1_pred, y1_target, h_x1, w2_train)
        dw1_x2, dw2_x2 = backward_numpy(x2, y2_pred, y2_target, h_x2, w2_train)
        dw1 = dw1_x1 + dw1_x2    # total gradient for all inputs
        dw2 = dw2_x1 + dw2_x2

        # update weights
        w1_train -= learning_rate * dw1
        w2_train -= learning_rate * dw2

    return w1_train, w2_train, losses

# training
w1_final_torch, w2_final_torch, losses_torch = train_torch()
w1_final_numpy, w2_final_numpy, losses_numpy = train_numpy()

# final result
print("\n학습 후 w1 가중치:")
print(f"PyTorch: \n{w1_final_torch}")
print(f"Numpy: \n{w1_final_numpy}")
print("\n학습 후 w2 가중치:")
print(f"PyTorch: \n{w2_final_torch}")
print(f"Numpy: \n{w2_final_numpy}")
