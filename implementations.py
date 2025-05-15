import numpy as np
from myutils import *


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using gradient descent.

    Parameters:
    y: np.ndarray - target values (n_samples, )
    tx: np.ndarray - feature matrix (n_samples, n_features)
    initial_w: np.ndarray - initial weights (n_features, )
    max_iters: int - maximum number of iterations
    gamma: float - step size for gradient descent

    Returns:
    w: np.ndarray - optimized weights after gradient descent
    loss: float - final loss value
    """
    w = initial_w
    for _ in range(max_iters):
        # Compute gradient and loss
        gradient = compute_gradient(y, tx, w)

        # Update w by gradient
        w -= gamma * gradient

        # print(
        #     "GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
        #         bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
        #     )
        # )

    loss = compute_loss(y, tx, w)

    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression using stochastic gradient descent.

    Parameters:
    y: np.ndarray - target values (n_samples, )
    tx: np.ndarray - feature matrix (n_samples, n_features)
    initial_w: np.ndarray - initial weights (n_features, )
    max_iters: int - maximum number of iterations
    gamma: float - step size for stochastic gradient descent

    Returns:
    w: np.ndarray - optimized weights after stochastic gradient descent
    loss: float - final loss value
    """
    w = initial_w
    for n_iter in range(max_iters):

        # Implement stochastic gradient descent
        n = np.random.randint(tx.shape[0])
        xn = np.array([tx[n, :]])
        yn = np.array([y[n]])
        gradient = compute_gradient(yn, xn, w)

        # Update w by gradient
        w -= gamma * gradient

        # print(
        #     "SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
        #         bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
        #     )
        # )

    loss = compute_loss(yn, xn, w)

    return w, loss


def least_squares(y, tx):
    """
    Least squares regression using normal equations.

    Parameters:
    y: np.ndarray - target values (n_samples, )
    tx: np.ndarray - feature matrix (n_samples, n_features)

    Returns:
    w: np.ndarray - optimized weights using normal equations
    loss: float - final loss value
    """
    A = tx.T @ tx
    b = tx.T @ y

    # Solve the system A @ w = b for w using np.linalg.solve (more stable than using np.linalg.inv)
    w = np.linalg.solve(A, b)

    # Compute the mean squared error
    loss = compute_loss(y, tx, w)

    return w, loss


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations.

    Parameters:
    y: np.ndarray - target values (n_samples, )
    tx: np.ndarray - feature matrix (n_samples, n_features)
    lambda_: float - regularization parameter

    Returns:
    w: np.ndarray - optimized weights after ridge regression
    loss: float - final loss value
    """
    D = tx.shape[1]  # Number of features

    # Create identity matrix of shape (D, D)
    I = np.identity(D)

    # Compute A = (tx^T * tx) + lambda * 2 * N * I
    A = tx.T @ tx + lambda_ * 2 * len(y) * I

    # Compute b = tx^T * y
    b = tx.T @ y

    # Solve A * w = b for w using np.linalg.solve
    w = np.linalg.solve(A, b)

    loss = compute_loss(y, tx, w)

    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent.

    Parameters:
    y: np.ndarray - binary target values (n_samples, )
    tx: np.ndarray - feature matrix (n_samples, n_features)
    initial_w: np.ndarray - initial weights (n_features, )
    max_iters: int - maximum number of iterations
    gamma: float - step size for gradient descent

    Returns:
    w: np.ndarray - optimized weights after gradient descent
    loss: float - final loss value
    """
    w = initial_w
    for _ in range(max_iters):
        pred = sigmoid(np.dot(tx, w))
        gradient = np.dot(tx.T, (pred - y)) / len(y)
        w -= gamma * gradient

    # Compute the loss
    pred = sigmoid(np.dot(tx, w))
    loss = -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))

    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent.

    Parameters:
    y: np.ndarray - binary target values (n_samples, )
    tx: np.ndarray - feature matrix (n_samples, n_features)
    lambda_: float - regularization parameter
    initial_w: np.ndarray - initial weights (n_features, )
    max_iters: int - maximum number of iterations
    gamma: float - step size for gradient descent

    Returns:
    w: np.ndarray - optimized weights after regularized logistic regression
    loss: float - final loss value
    """
    w = initial_w
    for _ in range(max_iters):
        pred = sigmoid(np.dot(tx, w))
        # L2 regularization
        gradient = np.dot(tx.T, (pred - y)) / len(y) + 2 * lambda_ * w
        w -= gamma * gradient

    # Compute the loss
    pred = sigmoid(np.dot(tx, w))
    loss = -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))

    return w, loss
