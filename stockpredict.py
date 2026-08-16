#LSTM stock prdiction example
import torch
import torch.nn as nn
import numpy as np

# -------------------------------
# Sample Stock Prices
# -------------------------------

stock_prices = np.array([
    100, 102, 104, 106, 108,
    110, 112, 115, 118, 120,
    123, 125, 128, 130, 133
], dtype=np.float32)

# -------------------------------
# Create Sequences
# -------------------------------

seq_length = 5

X = []
y = []

for i in range(len(stock_prices) - seq_length):
    X.append(stock_prices[i:i+seq_length])
    y.append(stock_prices[i+seq_length])

X = np.array(X)
y = np.array(y)

X = torch.tensor(X).unsqueeze(-1)
y = torch.tensor(y).unsqueeze(-1)

print("Input Shape :", X.shape)
print("Target Shape:", y.shape)

# -------------------------------
# LSTM Model
# -------------------------------

class StockLSTM(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=16,
            batch_first=True
        )

        self.fc = nn.Linear(16, 1)

    def forward(self, x):

        out, (hidden, cell) = self.lstm(x)

        out = out[:, -1, :]

        out = self.fc(out)

        return out

# -------------------------------
# Create Model
# -------------------------------

model = StockLSTM()

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)

# -------------------------------
# Training
# -------------------------------

epochs = 1000

for epoch in range(epochs):

    predictions = model(X)

    loss = criterion(predictions, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Loss: {loss.item():.4f}"
        )

# -------------------------------
# Print All Predictions
# -------------------------------

print("\nTraining Complete")
print("-" * 75)

with torch.no_grad():

    predictions = model(X)

    print(
        f"{'Input Sequence':<40}"
        f"{'Actual':<10}"
        f"{'Predicted':<10}"
    )

    print("-" * 75)

    for i in range(len(X)):

        sequence = X[i].squeeze().numpy()

        actual = y[i].item()

        predicted = predictions[i].item()

        print(
            f"{str(sequence):<40}"
            f"{actual:<10.2f}"
            f"{predicted:<10.2f}"
        )

# -------------------------------
# Future Prediction
# -------------------------------

future_input = torch.tensor(
    [[123, 125, 128, 130, 133]],
    dtype=torch.float32
).unsqueeze(-1)

with torch.no_grad():

    future_price = model(future_input)

print("\nFuture Prediction")
print("Input: [123,125,128,130,133]")
print(
    f"Predicted Next Stock Price: "
    f"{future_price.item():.2f}"
)