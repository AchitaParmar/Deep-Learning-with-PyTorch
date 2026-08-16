# RNN for rainfall prediction example
import torch
import torch.nn as nn
import numpy as np

# Rainfall data (mm)
rainfall = np.array([
    10, 12, 15, 18, 20,
    22, 25, 28, 30, 32,
    35, 37, 40, 42, 45
], dtype=np.float32)

# Create sequences
seq_length = 5

X = []
y = []

for i in range(len(rainfall) - seq_length):
    X.append(rainfall[i:i+seq_length])
    y.append(rainfall[i+seq_length])

X = torch.tensor(np.array(X)).unsqueeze(-1)
y = torch.tensor(np.array(y)).unsqueeze(-1)

# RNN Model
class RainfallRNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=16,
            batch_first=True
        )

        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        out, hidden = self.rnn(x)

        # Last output of sequence
        out = out[:, -1, :]

        out = self.fc(out)

        return out

# Create model
model = RainfallRNN()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train
epochs = 1000

for epoch in range(epochs):

    outputs = model(X)

    loss = criterion(outputs, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f}")

# ------------------------------------
# Print Predictions for all samples
# ------------------------------------

print("\nTraining Complete")
print("-" * 70)

with torch.no_grad():

    predictions = model(X)

    print(
        f"{'Input Sequence':<35}"
        f"{'Actual':<10}"
        f"{'Predicted':<10}"
    )

    print("-" * 70)

    for i in range(len(X)):

        sequence = X[i].squeeze().numpy()

        actual = y[i].item()

        predicted = predictions[i].item()

        print(
            f"{str(sequence):<35}"
            f"{actual:<10.2f}"
            f"{predicted:<10.2f}"
        )

# ------------------------------------
# Predict Future Rainfall
# ------------------------------------

future_input = torch.tensor(
    [[35, 37, 40, 42, 45]],
    dtype=torch.float32
).unsqueeze(-1)

with torch.no_grad():
    future_prediction = model(future_input)

print("\nFuture Prediction:")
print(f"Input: [35, 37, 40, 42, 45]")
print(f"Predicted Next Rainfall: {future_prediction.item():.2f} mm")