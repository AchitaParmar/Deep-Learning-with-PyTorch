#feed forward for house price prediction example
import torch
import torch.nn as nn
import numpy as np

# ----------------------------------
# Dataset
# ----------------------------------

# Features:
# [Area, Bedrooms, Age]

X = np.array([
    [1000, 2, 10],
    [1200, 2, 8],
    [1500, 3, 5],
    [1800, 3, 4],
    [2000, 4, 3],
    [2200, 4, 2],
    [2500, 5, 1]
], dtype=np.float32)

# House Prices (in lakhs)
y = np.array([
    [30],
    [35],
    [45],
    [55],
    [65],
    [72],
    [85]
], dtype=np.float32)

# Convert to tensors
X = torch.tensor(X)
y = torch.tensor(y)

# ----------------------------------
# Feed Forward Neural Network
# ----------------------------------

class HousePriceNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(3, 8)
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(8, 4)

        self.fc3 = nn.Linear(4, 1)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)

        return x

# Create model
model = HousePriceNN()

# Loss Function
criterion = nn.MSELoss()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# ----------------------------------
# Training
# ----------------------------------

epochs = 2000

for epoch in range(epochs):

    predictions = model(X)

    loss = criterion(predictions, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {loss.item():.4f}"
        )

# ----------------------------------
# Print Predictions
# ----------------------------------

print("\nTraining Complete")
print("-" * 60)

with torch.no_grad():

    predictions = model(X)

    print(
        f"{'Area':<10}"
        f"{'Beds':<10}"
        f"{'Age':<10}"
        f"{'Actual':<12}"
        f"{'Predicted':<12}"
    )

    print("-" * 60)

    for i in range(len(X)):

        area = X[i][0].item()
        beds = X[i][1].item()
        age = X[i][2].item()

        actual = y[i].item()
        predicted = predictions[i].item()

        print(
            f"{area:<10.0f}"
            f"{beds:<10.0f}"
            f"{age:<10.0f}"
            f"{actual:<12.2f}"
            f"{predicted:<12.2f}"
        )

# ----------------------------------
# Predict New House Price
# ----------------------------------

new_house = torch.tensor(
    [[1700, 3, 4]],
    dtype=torch.float32
)

with torch.no_grad():
    predicted_price = model(new_house)

print("\nNew House Prediction")
print("Area = 1700 sq ft")
print("Bedrooms = 3")
print("Age = 4 years")

print(
    f"Predicted Price = "
    f"{predicted_price.item():.2f} Lakhs"
)