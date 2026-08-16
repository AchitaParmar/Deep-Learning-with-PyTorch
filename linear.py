import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Sample Data
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]], dtype=np.float32)
y = np.array([[10], [20], [30], [40], [50], [60], [70], [80]], dtype=np.float32)

# Convert to tensors
X_train = torch.tensor(X)
y_train = torch.tensor(y)

# ANN Model for Regression
class RegressionANN(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.network(x)

# Create model
model = RegressionANN()

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
epochs = 1000

for epoch in range(epochs):

    predictions = model(X_train)

    loss = criterion(predictions, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Predict a new value
new_input = torch.tensor([[6.5]])

with torch.no_grad():
    predicted_value = model(new_input).item()

print(f"Predicted Value: {predicted_value:.2f}")

# Classification based on predicted value
if predicted_value < 40:
    category = "Low"
elif predicted_value < 70:
    category = "Medium"
else:
    category = "High"

print("Class:", category)