import numpy as np
from implementations import *


# -------------------------------- functions for implementations.py --------------------------------
def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    # Compute gradient vector

    e = y - np.dot(tx, w)

    gradient = -np.dot(tx.T, e) / len(y)

    return gradient


def compute_loss(y, tx, w):
    """
    Compute the MSE loss.
    """
    e = y - tx @ w
    loss = np.sum(e**2) / (2 * len(y))
    return loss


def sigmoid(x):
    """
    This function computes the sigmoid of the input 'x'.
    The sigmoid function maps any real value into the range (0, 1), often used
    in machine learning as an activation function.

    Args:
        x (int, float, list, np.array): Input value(s) to compute the sigmoid for.
                                        If 'x' is a list or np.array, the sigmoid will be computed element-wise.

    Returns:
        np.array: The computed sigmoid values, with the same shape as the input.
    """
    return 1 / (1 + np.exp(-np.array(x)))


# -------------------------------- functions for others --------------------------------
def undersample(filtered_xy_train, minority_class=1, majority_class=0):
    """
    Perform undersampling to balance the dataset based on the class labels.

    Parameters:
    filtered_xy_train (np.ndarray): Array where the last column represents labels (e.g., 1 and -1).
    minority_class (int): Label of the minority class that will be matched.
    majority_class (int): Label of the majority class to be undersampled.

    Returns:
    np.ndarray: The undersampled dataset with balanced classes.
    """
    # Identify the indices of the minority and majority classes
    minority_indices = np.where(filtered_xy_train[:, -1] == minority_class)[0]
    majority_indices = np.where(filtered_xy_train[:, -1] == majority_class)[0]

    # Randomly sample majority class to match the number of minority samples
    num_minority = len(minority_indices)
    undersampled_majority_indices = np.random.choice(
        majority_indices, size=num_minority, replace=False
    )

    # Combine the minority and sampled majority indices
    undersampled_indices = np.concatenate(
        [minority_indices, undersampled_majority_indices]
    )
    np.random.shuffle(undersampled_indices)

    # Create the undersampled dataset
    xy_train_undersampled = filtered_xy_train[undersampled_indices]

    return xy_train_undersampled


def oversample(filtered_xy_train, minority_class=1, majority_class=0):
    """
    Perform oversampling to balance the dataset by replicating minority class samples.

    Parameters:
    filtered_xy_train (np.ndarray): Array where the last column represents labels (e.g., 1 and -1).
    minority_class (int): Label of the minority class to be oversampled.
    majority_class (int): Label of the majority class.

    Returns:
    np.ndarray: The oversampled dataset with balanced classes.
    """
    # Identify the indices of the minority and majority classes
    minority_indices = np.where(filtered_xy_train[:, -1] == minority_class)[0]
    majority_indices = np.where(filtered_xy_train[:, -1] == majority_class)[0]

    # Calculate the number of samples needed to match majority samples
    num_majority = len(majority_indices)

    # Randomly sample with replacement from minority_indices to match the number of majority samples
    oversampled_minority_indices = np.random.choice(
        minority_indices, size=num_majority, replace=True
    )

    # Combine the original majority samples and the oversampled minority samples
    oversampled_indices = np.concatenate(
        [majority_indices, oversampled_minority_indices]
    )
    np.random.shuffle(oversampled_indices)

    # Create the oversampled dataset
    xy_train_oversampled = filtered_xy_train[oversampled_indices]

    return xy_train_oversampled


def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    # Avoid division by zero
    std[std == 0] = 1
    X_standardized = (X - mean) / std
    return X_standardized


def pca(X, num_components):
    """
    Perform Principal Component Analysis (PCA) on data X.

    Parameters:
    X: numpy array, shape (n_samples, n_features)
       The data matrix where each row is a sample and each column is a feature.
    num_components: int
       The number of principal components to return.

    Returns:
    X_pca: numpy array, shape (n_samples, num_components)
       The data projected onto the top 'num_components' principal components.
    """
    # Step 1: Center the data by subtracting the mean of each feature (column)
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean

    # Step 2: Calculate the covariance matrix of the centered data
    cov_matrix = np.cov(X_centered, rowvar=False)

    # Step 3: Compute the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Step 4: Sort the eigenvectors by decreasing eigenvalues (sort by importance of each component)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    sorted_eigenvalues = eigenvalues[sorted_indices]

    # Step 5: Select the top 'num_components' eigenvectors (principal components)
    eigenvector_subset = sorted_eigenvectors[:, :num_components]

    # Step 6: Project the data onto the new eigenvector basis
    X_pca = np.dot(X_centered, eigenvector_subset)

    return X_pca, sorted_eigenvalues, sorted_eigenvectors


def compute_accuracy(y_true, y_pred):
    """
    Compute the accuracy.
    """
    return np.mean(y_true == y_pred)


def compute_f1_score(y_true, y_pred, pos_label=1):
    """
    Compute the F1 score.
    """
    TP = np.sum((y_true == pos_label) & (y_pred == pos_label))
    FP = np.sum((y_true != pos_label) & (y_pred == pos_label))
    FN = np.sum((y_true == pos_label) & (y_pred != pos_label))

    if TP + FP + 1e-8 == 0 or TP + FN + 1e-8 == 0:
        return 0.0

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1
