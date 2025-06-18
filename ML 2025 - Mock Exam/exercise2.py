# Imports (add more from sklearn or torch if needed)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# Task 1: Load the dataset
data = pd.read_csv("./data/exercise2.csv")

#Task 2: Investigate data quality
# Investigate missing values
number_of_nans = data.isna().sum().sum()
print(f"Number of NaNs in the dataset: {number_of_nans}")

# Investigate outliers with a box-and-whisker plot
fig, ax = plt.subplots(figsize=(10, 5))
for col in data.columns:
    # TODO: create a box plot for each column (HINT: Search matplotlib documentation for the function to use)

plt.savefig("results/ex2_boxplot.png")
plt.close()

# Task 2: Split the dataset into training and test sets
X = # TODO: select feature matrix
y = # TODO: select target vector

X_train, X_test, y_train, y_test = # TODO: split the dataset in training/testing sets

# Task 3: Wrap the data for PyTorch training
batch_size = 256
train_dataset = #TODO: Move the training set to PyTorch tensors dataset
test_dataset_tensors = # TODO: Move the test set to PyTorch tensors dataset

train_loader = # TODO: create a DataLoader for the training set with batch_size
test_loader  = # TODO: create a DataLoader for the test set with batch_size

# Task 4: Define the model, loss function, and optimizer
class LinearReplicator(nn.Module):
    def __init__(self, in_features=9):
        super().__init__()
        # TODO: define the linear layer

    def forward(self, x):
        return self.lin(x)

model = LinearReplicator(in_features=X_train.shape[1])
criterion = # TODO: Define the loss function 
optimizer = # TODO: Define the optimizer
n_epochs = 200

for epoch in range(1, n_epochs + 1):
    
    model.train()
    epoch_loss = 0.0
    
    for xb, yb in train_loader:
        
        # TODO: Complete the loop to perform the usual training steps. WARNING: You need to 
        
    epoch_loss /= len(train_loader.dataset)

    if epoch == 1 or epoch % 50 == 0:
        print(f"Epoch {epoch:3d} — Train MSE: {epoch_loss:.4f}")

# Task 5: Evaluate on the test set and plot predictions vs. true values
model.eval()
with torch.no_grad():
    all_preds = []
    all_targets = []
    for xb, yb in test_loader:
        #TODO: make the prediction for the observation on the test set and add the target and the prediction to the appropriate list
    y_hat  = torch.vstack(all_preds)
    y_true = torch.vstack(all_targets)

mse_test = criterion(y_hat, y_true).item()
print(f"\nTest set MSE: {mse_test:.4f}")

# Scatter plot of true vs. predicted
# TODO: Create a scatter plot that plot the y_true vs. y_pred. (HINT: Search matplotlib documentation for the function to use)