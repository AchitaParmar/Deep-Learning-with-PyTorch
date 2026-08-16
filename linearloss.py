import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

print("Program Started...\n")

# Dataset
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]], dtype=np.float32)
y = np.array([[10], [20], [30], [40], [50], [60], [70], [80]], dtype=np.float32)

# Convert to tensors
X_train = torch.tensor(X)
y_train = torch.tensor(y)

# ANN Model
class RegressionANN(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.network(x)

# Loss Functions
loss_functions = {
    "MSELoss": nn.MSELoss(),
    "L1Loss": nn.L1Loss(),
    "HuberLoss": nn.HuberLoss(),
    "SmoothL1Loss": nn.SmoothL1Loss()
}

epochs = 1000
results = []

for loss_name, criterion in loss_functions.items():

    print(f"Training with {loss_name}...")

    # New model for each loss function
    model = RegressionANN()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training Loop
    for epoch in range(epochs):

        predictions = model(X_train)

        loss = criterion(predictions, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Test Prediction
    test_input = torch.tensor([[6.5]])

    with torch.no_grad():
        predicted_value = model(test_input).item()

    # Classification
    if predicted_value < 40:
        category = "Low"
    elif predicted_value < 70:
        category = "Medium"
    else:
        category = "High"

    results.append([
        loss_name,
        round(loss.item(), 6),
        round(predicted_value, 2),
        category
    ])

# Display Results
print("\n")
print("=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

print(f"{'Loss Function':<20} {'Final Loss':<15} {'Prediction':<15} {'Class'}")
print("-" * 70)

for result in results:
    print(
        f"{result[0]:<20} "
        f"{result[1]:<15} "
        f"{result[2]:<15} "
        f"{result[3]}"
    )

print("\nProgram Finished Successfully!")