import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Dataset
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]], dtype=np.float32)
y = np.array([[10], [20], [30], [40], [50], [60], [70], [80]], dtype=np.float32)

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

# List of optimizers to test
optimizers = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
    "RMSprop": optim.RMSprop,
    "Adagrad": optim.Adagrad
}

criterion = nn.MSELoss()
epochs = 1000

results = []

for opt_name, opt_class in optimizers.items():

    print(f"\n{'='*50}")
    print(f"Training with {opt_name}")
    print(f"{'='*50}")

    # Create a NEW model for each optimizer
    model = RegressionANN()

    # Create optimizer
    optimizer = opt_class(model.parameters(), lr=0.01)

    # Training
    for epoch in range(epochs):

        predictions = model(X_train)

        loss = criterion(predictions, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Test prediction
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
        opt_name,
        round(loss.item(), 6),
        round(predicted_value, 2),
        category
    ])

# Display results
print("\n\nFINAL COMPARISON")
print("-" * 70)
print(f"{'Optimizer':<12} {'Loss':<15} {'Prediction':<15} {'Class'}")
print("-" * 70)

for result in results:
    print(f"{result[0]:<12} {result[1]:<15} {result[2]:<15} {result[3]}")