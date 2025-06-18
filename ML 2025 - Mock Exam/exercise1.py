import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

np.random.seed(10)

# 1. Load the data
df = pd.read_csv('./data/exercise1.csv')
X = df.iloc[:, :7].values  # first 7 columns as features
y = df.iloc[:, 7].values   # last column as target

# 2. Split into train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 3. Fit a linear model on the 7 features
lr = LinearRegression()
lr.fit(X_train, y_train)
y_train_pred_lr = lr.predict(X_train)
y_test_pred_lr = lr.predict(X_test)

mse_train_lr = mean_squared_error(y_train, y_train_pred_lr)
mse_test_lr = mean_squared_error(y_test, y_test_pred_lr)

print(f"Linear Regression (7 features) - In-sample MSE: {mse_train_lr:.4f}")
print(f"Linear Regression (7 features) - Out-of-sample MSE: {mse_test_lr:.4f}\n")

# 4. Prepare random feature generation
def relu(x):
    return np.maximum(0, x)

ks = [1, 10, 20, 50, 75, 100, 1000]
alphas = [0.1, 1, 10]
results = []

for k in ks:
    # Generate random weights W ∼ N(0,1) of shape (7, k)
    W = np.random.randn(X_train.shape[1], k)
    # Create random features
    X_train_rf = relu(X_train @ W)
    X_test_rf = relu(X_test @ W)
    
    # Augment original features with RFs
    X_train_aug = np.hstack([X_train, X_train_rf])
    X_test_aug = np.hstack([X_test, X_test_rf])
    
    for alpha in alphas:
        # 5. Fit Ridge regression
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train_aug, y_train)
        
        y_train_pred = ridge.predict(X_train_aug)
        y_test_pred = ridge.predict(X_test_aug)
        
        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)
        
        results.append({
            'k': k,
            'alpha': alpha,
            'mse_train': mse_train,
            'mse_test': mse_test
        })

# Summarize results
results_df = pd.DataFrame(results)
print("Ridge Regression Results (with Random Features):")
print(results_df.pivot(index='k', columns='alpha', values=['mse_train', 'mse_test']))

# 1) Pivot your results_df into a (k × α) table of test MSE
pivot_mse = results_df.pivot(index='k', columns='alpha', values='mse_test')

# 2) Convert to RMSE
pivot_rmse = np.sqrt(pivot_mse)

# 3) Plot heatmap
fig, ax = plt.subplots()
cax = ax.imshow(pivot_rmse.values, aspect='auto')  # default colormap

# 4) Label the ticks
ax.set_xticks(np.arange(len(pivot_rmse.columns)))
ax.set_xticklabels(pivot_rmse.columns)
ax.set_yticks(np.arange(len(pivot_rmse.index)))
ax.set_yticklabels(pivot_rmse.index)

ax.set_xlabel('Ridge Regularization (α)')
ax.set_ylabel('Number of Random Features (k)')
ax.set_title('Test RMSE Heatmap')

# 5) Annotate each cell with its RMSE
for i in range(pivot_rmse.shape[0]):
    for j in range(pivot_rmse.shape[1]):
        ax.text(j, i, f"{pivot_rmse.values[i, j]:.3f}",
                ha='center', va='center')

# 6) Add colorbar and tidy up
fig.colorbar(cax, ax=ax, label='RMSE')
plt.tight_layout()
plt.show()
