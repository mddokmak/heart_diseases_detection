import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

torch.manual_seed(42)

# Load CSV
df = pd.read_csv('data/exercise2.csv')

# Columns:
# 0: S_T
# 1–4: call payoffs at strikes K1…K4
# 5–8: put payoffs at strikes K5…K8
# 9: f(S_T)

#! There is a typo in the question: The target is the second column, not the last one. Apologies for the error, but that doesn't change anything to the rest of the exercise.
X = np.concatenate((df.iloc[:, 0:1].to_numpy().reshape(-1,1), df.iloc[:, 2:].to_numpy()), axis = 1).astype(np.float32)
y = df.iloc[:, 1].values.astype(np.float32).reshape(-1,1)

# Investigate outliers with a box-and-whisker plot
fig, ax = plt.subplots(figsize=(10, 5))
count = 0
for col in df.columns:
    ax.boxplot(df[col], positions=[count], widths=0.5, vert=False, patch_artist=True)
    count += 1
plt.show()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Wrap in DataLoader
batch_size = 256
train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
                          batch_size=batch_size)

class LinearReplicator(nn.Module):
    def __init__(self, in_features=9):
        super().__init__()
        # one linear layer: weights on [S_T, calls…, puts…], bias = bond
        self.lin = nn.Linear(in_features, 1, bias=True)
    def forward(self, x):
        return self.lin(x)
    
model = LinearReplicator(in_features=9)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1)
n_epochs = 200

for epoch in range(1, n_epochs+1):
    model.train()
    epoch_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        y_pred = model(xb)
        loss = criterion(y_pred, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)
    epoch_loss /= len(train_loader.dataset)

    if epoch % 50 == 0 or epoch == 1:
        print(f'Epoch {epoch:3d} — Train MSE: {epoch_loss:.4f}')

model.eval()
with torch.no_grad():
    # compute on test set
    X_all = torch.from_numpy(X_test)
    y_true = torch.from_numpy(y_test)
    y_hat  = model(X_all)

mse_test = criterion(y_hat, y_true).item()
print(f'\nTest set MSE: {mse_test:.4f}')

# scatter true vs pred
plt.figure(figsize=(6,6))
plt.scatter(y_true.numpy(), y_hat.numpy(), alpha=0.3, s=10)
plt.plot([y_true.min(), y_true.max()],
         [y_true.min(), y_true.max()],
         'k--', lw=1)
plt.xlabel('True f(S_T)')
plt.ylabel('Predicted f̂(S_T)')
plt.title('Replication: True vs. Predicted')
plt.tight_layout()
plt.show()
