from matplotlib import pyplot as plt
from implementations import *
from myutils import *
from svm import *


def build_poly(x, degree):
    """Build polynomial basis functions for input data x up to the given degree."""
    if degree == 0:
        return x
    else:
        poly = np.ones((x.shape[0], 1))  # Start with bias term
        for deg in range(1, degree + 1):
            poly = np.hstack((poly, x**deg))
        return poly


def build_k_indices(y, k_fold, seed=1):
    num_rows = y.shape[0]
    interval = int(num_rows / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_rows)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def k_fold_cross_validation(y, x, k_indices, k, method, **hyperparams):
    degree = hyperparams.get("degree", 0)  # Default degree is 0
    test_indices = k_indices[k]
    train_indices = np.delete(k_indices, k, axis=0).reshape(-1)
    x_train = x[train_indices]
    x_test = x[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    # Build polynomial features
    x_train_poly = build_poly(x_train, degree)
    x_test_poly = build_poly(x_test, degree)

    if method == "mean_squared_error_gd":
        gamma = hyperparams["gamma"]
        max_iters = hyperparams["max_iters"]
        initial_w = np.zeros(x_train_poly.shape[1])
        w, loss = mean_squared_error_gd(
            y_train, x_train_poly, initial_w, max_iters, gamma
        )
        y_pred = x_test_poly @ w
        y_pred_labels = np.where(y_pred >= 0.5, 1, -1)
    elif method == "mean_squared_error_sgd":
        gamma = hyperparams["gamma"]
        max_iters = hyperparams["max_iters"]
        initial_w = np.zeros(x_train_poly.shape[1])
        w, loss = mean_squared_error_sgd(
            y_train, x_train_poly, initial_w, max_iters, gamma
        )
        y_pred = x_test_poly @ w
        y_pred_labels = np.where(y_pred >= 0.5, 1, -1)
    elif method == "least_squares":
        w, loss = least_squares(y_train, x_train_poly)
        y_pred = x_test_poly @ w
        y_pred_labels = np.where(y_pred >= 0.5, 1, -1)
    elif method == "ridge_regression":
        lambda_ = hyperparams["lambda_"]
        w, loss = ridge_regression(y_train, x_train_poly, lambda_)
        y_pred = x_test_poly @ w
        y_pred_labels = np.where(y_pred >= 0.5, 1, -1)
    elif method == "logistic_regression":
        gamma = hyperparams["gamma"]
        max_iters = hyperparams["max_iters"]
        initial_w = np.zeros(x_train_poly.shape[1])
        y_train_binary = np.where(y_train == -1, 0, 1)
        w, loss = logistic_regression(
            y_train_binary, x_train_poly, initial_w, max_iters, gamma
        )
        y_pred_prob = sigmoid(x_test_poly @ w)
        y_pred_labels = np.where(y_pred_prob >= 0.5, 1, -1)
    elif method == "reg_logistic_regression":
        gamma = hyperparams["gamma"]
        max_iters = hyperparams["max_iters"]
        lambda_ = hyperparams["lambda_"]
        initial_w = np.zeros(x_train_poly.shape[1])
        y_train_binary = np.where(y_train == -1, 0, 1)
        w, loss = reg_logistic_regression(
            y_train_binary, x_train_poly, lambda_, initial_w, max_iters, gamma
        )
        y_pred_prob = sigmoid(x_test_poly @ w)
        y_pred_labels = np.where(y_pred_prob >= 0.5, 1, -1)
    elif method == "svm":
        C = hyperparams["C"]
        learning_rate = hyperparams["learning_rate"]
        n_iters = hyperparams["n_iters"]
        w, b, loss_history = train_svm(
            x_train_poly, y_train, C=C, learning_rate=learning_rate, n_iters=n_iters
        )
        y_pred_labels = predict(x_test_poly, w, b)
        loss = loss_history[-1]
    else:
        raise ValueError(f"Unknown method: {method}")

    # Compute metrics
    acc = compute_accuracy(y_test, y_pred_labels)
    f1 = compute_f1_score(y_test, y_pred_labels, pos_label=1)

    return acc, f1, loss


def tune_hyperparameters(method, y, x, k_indices, k_fold):
    hyperparams_list = generate_hyperparameters(method)
    best_hyperparams = None
    best_f1_score = -np.inf
    best_metrics = None  # To store accuracy, f1_score, loss
    for hyperparams in hyperparams_list:
        accuracies = []
        f1_scores = []
        losses = []
        for k in range(k_fold):
            acc, f1, loss = k_fold_cross_validation(
                y, x, k_indices, k, method, **hyperparams
            )
            accuracies.append(acc)
            f1_scores.append(f1)
            losses.append(loss)
        avg_accuracy = np.mean(accuracies)
        avg_f1_score = np.mean(f1_scores)
        avg_loss = np.mean(losses)
        if avg_f1_score > best_f1_score:
            best_f1_score = avg_f1_score
            best_hyperparams = hyperparams
            best_metrics = {
                "accuracy": avg_accuracy,
                "f1_score": avg_f1_score,
                "loss": avg_loss,
            }
    return best_hyperparams, best_metrics


def generate_hyperparameters(method):
    hyperparams = []
    degrees = [0, 1, 2, 3, 4, 5]
    if method == "mean_squared_error_gd":
        gammas = [1e-4, 1e-3, 1e-2, 1e-1]
        max_iters_list = [50, 100, 200, 300]
        for degree in degrees:
            for gamma in gammas:
                for max_iters in max_iters_list:
                    hyperparams.append(
                        {"degree": degree, "gamma": gamma, "max_iters": max_iters}
                    )
    elif method == "mean_squared_error_sgd":
        gammas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
        max_iters_list = [100, 500, 1000, 5000]
        for degree in degrees:
            for gamma in gammas:
                for max_iters in max_iters_list:
                    hyperparams.append(
                        {"degree": degree, "gamma": gamma, "max_iters": max_iters}
                    )
    elif method == "least_squares":
        for degree in degrees:
            hyperparams.append({"degree": degree})
    elif method == "ridge_regression":
        lambdas = np.logspace(-4, 0, 50)
        for degree in degrees:
            for lambda_ in lambdas:
                hyperparams.append({"degree": degree, "lambda_": lambda_})
    elif method == "logistic_regression":
        gammas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
        max_iters_list = [100, 500, 1000, 5000]
        for degree in degrees:
            for gamma in gammas:
                for max_iters in max_iters_list:
                    hyperparams.append(
                        {"degree": degree, "gamma": gamma, "max_iters": max_iters}
                    )
    elif method == "reg_logistic_regression":
        gammas = [1e-5, 1e-4, 1e-3, 1e-2]
        max_iters_list = [100, 500, 1000, 5000]
        lambdas = [1e-5, 1e-4, 1e-3, 1e-2]
        for degree in degrees:
            for gamma in gammas:
                for max_iters in max_iters_list:
                    for lambda_ in lambdas:
                        hyperparams.append(
                            {
                                "degree": degree,
                                "gamma": gamma,
                                "max_iters": max_iters,
                                "lambda_": lambda_,
                            }
                        )
    elif method == "svm":
        C_values = [0.01, 0.1, 1.0, 10.0]
        learning_rates = [1e-4, 1e-3, 1e-2, 1e-1]
        n_iters_list = [25, 50, 100, 200]
        for degree in degrees:
            for C in C_values:
                for learning_rate in learning_rates:
                    for n_iters in n_iters_list:
                        hyperparams.append(
                            {
                                "degree": degree,
                                "C": C,
                                "learning_rate": learning_rate,
                                "n_iters": n_iters,
                            }
                        )
    else:
        raise ValueError(f"Unknown method: {method}")
    return hyperparams


def plot_ridge_regression_performance(y, x, degree):
    # lambdas we want to try in the cross-validation
    lambdas = np.logspace(-4, 0, 50)
    # intialize lists to store the accuracies and the f1 scores
    accuracies = []
    f1_scores = []
    x_poly = build_poly(x, degree)
    for lambda_ in lambdas:
        w, loss = ridge_regression(y, x_poly, lambda_)
        y_pred = x_poly @ w
        y_pred_labels = np.where(y_pred >= 0.5, 1, 0)
        acc = compute_accuracy(y, y_pred_labels)
        f1 = compute_f1_score(y, y_pred_labels)
        f1_scores.append(f1)
        accuracies.append(acc)
    # getting the index of the lambda that gives the max f1 score
    max_f1_index = np.argmax(f1_scores)
    max_f1_lambda = lambdas[max_f1_index]
    # plot
    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, accuracies, marker="o", label="Accuracy")
    plt.plot(lambdas, f1_scores, marker="o", label="F1 Score")
    plt.axvline(
        x=max_f1_lambda,
        color="r",
        linestyle="--",
        label=f"Max F1 Score (λ={max_f1_lambda:.4f})",
    )
    plt.xscale("log")
    plt.xlabel("Lambda")
    plt.ylabel("Score")
    plt.title(
        "Cross-Validation Example: Accuracy and F1 Score vs Lambda for Ridge Regression (Degree 3)"
    )
    plt.legend()
    plt.show()
