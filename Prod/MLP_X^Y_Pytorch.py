import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

# I. Define MLP
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            # nn.ReLU(), -> last hidden layer linear
            nn.Linear(32, 1)   # identity output
        )
    def forward(self, x):
        return self.net(x)

model = MLP()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # much better with this
# optimizer = optim.SGD(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# II. Training data

# Generate training data 1
samples_1 = [
    ([1,2], [1]),
    ([2,2], [4]),
    ([3,2], [9]),
    ([4,2], [16]),
    ([5,2], [25]),
    ([6,2], [36]),
    ([7,2], [49]),
    ([8,2], [64]),
    ([9,2], [81]),
    ([10,2], [100]),

    ([2,3], [8]),
    ([3,3], [27]),
    ([4,3], [64]),
    ([5,3], [125]),
    ([6,3], [216]),
    ([7,3], [343]),
    ([8,3], [512]),
    ([9,3], [729]),
    ([10,3], [1000]),

    ([2,4], [16]),
    ([3,4], [81]),
    #([4,4], [256]),
    ([5,4], [625]),

    ([2,5], [32]),
    ([3,5], [243]),
    ([4,5], [1024]),  

    ([2,6], [64]),
    ([3,6], [729]),

    ([2,7], [128]),
    ([3,7], [2187]),  

    ([2,8], [256]),
    ([2,9], [512]),
    ([2,10], [1024]), 

    ([3,1], [3]),
    ([4,1], [4]),
    ([5,1], [5]),
    ([6,1], [6]),
    ([7,1], [7]),
    ([8,1], [8]),
    ([9,1], [9]),
    ([10,1], [10])
]

# For vectorized Numpy
X_np = np.array([x for x, y in samples_1], dtype=np.float32)
Y_np = np.array([np.log(y) for x, y in samples_1], dtype=np.float32)

X_train_1 = torch.tensor(X_np, dtype=torch.float32)
Y_train_1 = torch.tensor(Y_np, dtype=torch.float32) 

# for the Python loop
samples_1 = list(zip(X_train_1, Y_train_1))
# or
# samples_1 = [
#     (torch.tensor(x, dtype=torch.float32),
#      torch.tensor([np.log(y[0])], dtype=torch.float32))
#     for x, y in samples_1
# ]


# Generate training data 2
X_2 = np.random.uniform(0.001, 20.0, (10000, 2))   
Y_2 = np.power(X_2[:,0], X_2[:,1])                  # target: x^y

# For vectorized numpy
X_train_2 = torch.tensor(X_2, dtype=torch.float32)
Y_train_2 = torch.tensor(np.log(Y_2), dtype=torch.float32).unsqueeze(1)

# for the Python loop
# samples_2 = [(X_train_2[i], Y_train_2[i]) for i in range(len(X_train_2))]
samples_2 = list(zip(X_train_2, Y_train_2))


# III. Training methods

# Training with Python loop
def train_python_loop(model, optimizer, loss_fn, samples, epochs=5000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        random.shuffle(samples)
        total_loss = 0.0 
        
        for x, y in samples:
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            total_loss = total_loss + loss # keep this format
    
        total_loss /= len(samples)
        total_loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss {total_loss.item():.5f}")


# Training with vectorized Numpy (Better)
def train_vectorized(model, optimizer, loss_fn, X_train, Y_train, epochs=25000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        Y_pred = model(X_train)
        loss = loss_fn(Y_pred, Y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss {loss.item():.5f}")


# IV. Test
train_python_loop(model, optimizer, loss_fn, samples_1, epochs=5000);
# train_python_loop(model, optimizer, loss_fn, samples_2, epochs=1000);

test = torch.tensor([[2.0, 3.0]], dtype=torch.float32)  # 2^3 = 8
print("Prediction 2^3:", np.exp(model(test).item()))
print("Prediction (3, 3):", np.exp(model( torch.tensor([[3.0, 3.0]], dtype=torch.float32) ).item()))
print("Prediction (10, 2):", np.exp(model( torch.tensor([[10.0, 2.0]], dtype=torch.float32) ).item()))
print("Prediction (7, 2):", np.exp(model( torch.tensor([[7.0, 2.0]], dtype=torch.float32) ).item()))
print("Prediction (4, 4):", np.exp(model( torch.tensor([[4.0, 4.0]], dtype=torch.float32) ).item()))
print("Prediction (2, 11):", np.exp(model( torch.tensor([[2.0, 11.0]], dtype=torch.float32) ).item()))
print("Prediction (10, 5):", np.exp(model( torch.tensor([[10.0, 5.0]], dtype=torch.float32) ).item()))


#New MLP
model_2 = MLP()
optimizer_2 = optim.Adam(model_2.parameters(), lr=0.001)  # much better with this
# optimizer_2 = optim.SGD(model_2.parameters(), lr=0.001)
#train_vectorized(model_2, optimizer_2, loss_fn, X_train_1, Y_train_1, epochs=50000);
train_vectorized(model_2, optimizer_2, loss_fn, X_train_2, Y_train_2, epochs=50000);

test = torch.tensor([[2.0, 3.0]], dtype=torch.float32)  # 2^3 = 8
print("Prediction 2^3:", np.exp(model_2(test).item()))
print("Prediction (3, 3):", np.exp(model_2( torch.tensor([[3.0, 3.0]], dtype=torch.float32) ).item()))
print("Prediction (10, 2):", np.exp(model_2( torch.tensor([[10.0, 2.0]], dtype=torch.float32) ).item()))
print("Prediction (7, 2):", np.exp(model_2( torch.tensor([[7.0, 2.0]], dtype=torch.float32) ).item()))
print("Prediction (4, 4):", np.exp(model_2( torch.tensor([[4.0, 4.0]], dtype=torch.float32) ).item()))
print("Prediction (2, 11):", np.exp(model_2( torch.tensor([[2.0, 11.0]], dtype=torch.float32) ).item()))
print("Prediction (10, 5):", np.exp(model_2( torch.tensor([[10.0, 5.0]], dtype=torch.float32) ).item()))