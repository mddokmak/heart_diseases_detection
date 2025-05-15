import numpy as np


def initialize_parameters(n_features):
    w = np.zeros(n_features)
    b = 0.0
    return w, b


def compute_loss_and_gradient(w, b, X, y, C):
    n_samples = X.shape[0]
    distances = 1 - y * (np.dot(X, w) + b)
    distances = np.maximum(0, distances)  # Hinge loss

    # Compute loss
    hinge_loss = C * np.mean(distances)
    loss = 0.5 * np.dot(w, w) + hinge_loss

    # Compute gradient
    dw = w.copy()
    db = 0.0
    # Only consider points where hinge loss is non-zero
    mask = distances > 0
    if np.any(mask):
        y_masked = y[mask]
        X_masked = X[mask]
        dw -= C * np.dot(X_masked.T, y_masked) / n_samples
        db -= C * np.sum(y_masked) / n_samples

    return loss, dw, db


def train_svm(X, y, C=1.0, learning_rate=0.001, n_iters=1000):
    n_samples, n_features = X.shape
    w, b = initialize_parameters(n_features)
    loss_history = []

    for it in range(1, n_iters + 1):
        loss, dw, db = compute_loss_and_gradient(w, b, X, y, C)

        # Update parameters
        w -= learning_rate * dw
        b -= learning_rate * db

        if it % 100 == 0 or it == 1:
            loss_history.append(loss)
            # print(f"Iteration {it}/{n_iters}, Loss: {loss:.4f}")

    return w, b, loss_history


def predict(X, w, b):
    predictions = np.dot(X, w) + b
    return np.sign(predictions)


# def accuracy_score(y_true, y_pred):
#     correct = np.sum(y_true == y_pred)
#     return correct / len(y_true)

# def compute_metrics(y_true, y_pred):
#     TP = np.sum((y_true == 1) & (y_pred == 1))
#     TN = np.sum((y_true == -1) & (y_pred == -1))
#     FP = np.sum((y_true == -1) & (y_pred == 1))
#     FN = np.sum((y_true == 1) & (y_pred == -1))

#     precision = TP / (TP + FP + 1e-8)
#     recall = TP / (TP + FN + 1e-8)
#     f1 = 2 * precision * recall / (precision + recall + 1e-8)

#     return precision, recall, f1
