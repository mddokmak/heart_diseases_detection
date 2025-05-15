import numpy as np


def remove_empty_columns(data, labels, feature_mask):
    """
    Updates the feature_mask by marking columns with all NaN values in any class for removal.
    """
    # Apply the current feature mask to data
    data_masked = data[:, feature_mask]

    # Check for columns with all NaN values in each class
    all_nan_class_1 = np.all(np.isnan(data_masked[labels == 1]), axis=0)
    all_nan_class_minus1 = np.all(np.isnan(data_masked[labels == 0]), axis=0)

    # Identify columns to remove
    columns_to_remove_in_masked = np.where(all_nan_class_1 | all_nan_class_minus1)[0]

    # Map back to original feature indices
    masked_indices = np.where(feature_mask)[0]
    features_to_remove = masked_indices[columns_to_remove_in_masked]

    # Update feature_mask
    feature_mask[features_to_remove] = False

    return feature_mask, features_to_remove


def replace_nan_with_class_mean(data, labels):
    """
    Replaces NaN values in each feature column with the mean of that feature, calculated separately for each class.
    """
    for cls in np.unique(labels):
        # Create a mask for the current class
        class_mask = labels == cls

        # Calculate column-wise means for the current class, ignoring NaNs
        class_means = np.nanmean(data[class_mask], axis=0)

        # Replace NaNs within this class
        nan_indices = np.where(np.isnan(data) & class_mask[:, None])
        data[nan_indices] = np.take(class_means, nan_indices[1])

    print("Replaced NaN values with the mean of the feature of the respective class.")
    return data


def replace_nan_with_column_mean(data):
    """
    Replaces NaN values in each feature column with the mean of that feature.
    """
    # Calculate column-wise means for the current class, ignoring NaNs
    class_means = np.nanmean(data, axis=0)

    # Replace NaNs within this class
    nan_indices = np.where(np.isnan(data))
    data[nan_indices] = np.take(class_means, nan_indices[1])

    print("Replaced NaN values with the mean of the feature.")
    return data


def calculate_skewness(column):
    """
    Calculates skewness for a given column, ignoring NaN values.
    """
    n = len(column)
    mean = np.nanmean(column)
    std_dev = np.nanstd(column)

    if std_dev == 0:
        return 0

    skewness = (n / ((n - 1) * (n - 2))) * np.nansum(((column - mean) / std_dev) ** 3)
    return skewness


def log_scale_skewed_features(data, skew_threshold=1.0):
    """
    Identifies columns with skewness above the threshold and applies log scaling to these features.
    """
    # Exclude the target column (assuming it's the last column, e.g., class labels)
    feature_columns = range(data.shape[1] - 1)
    skewed_columns = []

    # Identify skewed columns
    for col in feature_columns:
        skew_val = calculate_skewness(data[:, col])
        if abs(skew_val) > skew_threshold:
            skewed_columns.append(col)

    # Apply log scaling to identified skewed columns
    for col in skewed_columns:
        data[:, col] = np.log1p(data[:, col])  # log1p handles zero values safely

    return data


def remove_features_with_low_class_separation(
    x_train_data, y_train_data, feature_mask, threshold=0.05
):
    """
    Updates the feature_mask by marking features with low class separation for deletion.
    """

    # Apply the current feature mask to x_train_data
    x_train_data_masked = x_train_data[:, feature_mask]

    # Calculate the mean for each feature by class, ignoring NaNs
    class_1_mean = np.nanmean(x_train_data_masked[y_train_data == 1], axis=0)
    class_minus1_mean = np.nanmean(x_train_data_masked[y_train_data == -1], axis=0)

    # Compute mean differences
    mean_diff = np.abs(class_1_mean - class_minus1_mean)

    # Identify features with mean difference lower than the threshold (low class separation)
    low_separation = mean_diff < threshold

    # Get indices of features currently True in the feature_mask
    masked_indices = np.where(feature_mask)[0]

    # Update feature_mask in-place: set to False where low_separation is True
    features_to_remove = masked_indices[low_separation]
    feature_mask[features_to_remove] = False

    # Return the updated feature_mask
    return feature_mask, features_to_remove


def remove_features_with_low_variance(x_train_data, feature_mask, threshold=0.05):
    """
    Updates the feature_mask by marking features with low variance for removal.
    """
    # Apply the current feature mask to x_train_data
    x_train_data_masked = x_train_data[:, feature_mask]

    # Calculate variance across all samples for each feature, ignoring NaNs
    variances = np.nanvar(x_train_data_masked, axis=0)

    # Identify features with variance below the threshold
    low_variance = variances < threshold

    # Get indices of features currently True in the feature_mask
    masked_indices = np.where(feature_mask)[0]

    # Features to remove
    features_to_remove = masked_indices[low_variance]

    # Update feature_mask in-place
    feature_mask[features_to_remove] = False

    return feature_mask, features_to_remove


def remove_highly_correlated_features(
    x_train_data, feature_mask, correlation_threshold=0.8
):
    """
    Updates the feature_mask by marking features that are highly correlated with others for removal.
    """
    # Apply the current feature mask to x_train_data
    x_train_data_masked = x_train_data[:, feature_mask]

    # Calculate correlation matrix for x_train_data_masked
    correlation_matrix = np.corrcoef(x_train_data_masked, rowvar=False)

    # Set threshold for high correlation
    corr_threshold = correlation_threshold

    # Find pairs of highly correlated features
    num_features = correlation_matrix.shape[0]
    high_correlation_pairs = []
    for i in range(num_features):
        for j in range(i + 1, num_features):
            if abs(correlation_matrix[i, j]) > corr_threshold:
                high_correlation_pairs.append((i, j))

    print(
        "Highly correlated feature pairs (indices in masked data):",
        high_correlation_pairs,
    )

    # Identify features to remove
    features_to_remove_in_masked = set()
    for i, j in high_correlation_pairs:
        features_to_remove_in_masked.add(j)  # Remove the second feature in each pair

    # Map back to original feature indices
    masked_indices = np.where(feature_mask)[0]
    features_to_remove = masked_indices[list(features_to_remove_in_masked)]

    # Update feature_mask in-place
    feature_mask[features_to_remove] = False

    return feature_mask, features_to_remove


def pearson(x, y):
    """
    This function is to determine the correlation coefficient of each feature with the output
    Args:
        x: array containing all data
        y: array containing output
    Returns:
        list[float]: Pearson correlation coefficient of each feature with the output.
    """
    list_pearson_coef = []
    for attribute in range(x.shape[1]):
        c = x[:, attribute]
        corr_coef = np.corrcoef(c, y)[0, 1]
        list_pearson_coef.append(corr_coef)
    return list_pearson_coef


def count_unique_values(x):
    """
    This function is to determine the number of unique values of each attribute,
    to determine if it is a categorical feature or a continuous one
    Args:
        x: array containing all data
        y: array containing output
    Returns:
        list[int]: number of unique values of each feature
    """
    list_unique_value = []
    for attribute in range(x.shape[1]):
        c = x[:, attribute]
        list_unique_value.append(len(np.unique(c)))
    return list_unique_value


def feature_selection(x, y, max_unique_values, min_corr_coef):
    """
    Keep only the features that satisfies our count constraints.
    Args:
        x: The dataset
        y: The output
        max_unique_values:maximum number of unique values to still be considered as a categorical feature.
        min_correlation_coef:The minimum correlation coefficient with the output to still be considered as a relevant feature.
    Returns: The indices of features that satisfy both constraints.
    """
    list_unique_value = count_unique_values(x)
    list_pearson_coef = pearson(x, y)
    # Features that we consider as categorical
    categorical_index = np.where(np.array(list_unique_value) < max_unique_values)[0]
    categorical_index = categorical_index.tolist()
    # Features that has a sufficient high correlation with the output
    correlated_values_index = np.where(np.array(list_pearson_coef) > min_corr_coef)[0]
    correlated_values_index = correlated_values_index.tolist()
    # Features that satisfy both constraints, categorical and correlated with the output
    correlated_categories = np.intersect1d(categorical_index, correlated_values_index)

    print(
        f"Features with less than {max_unique_values} unique values : {len(categorical_index)}\n"
        f"\t{categorical_index}\n\n"
        f"Features with more than {min_corr_coef} of correlation coefficient : {len(correlated_values_index)}\n"
        f"\t{correlated_values_index}\n\n"
        f"Features respecting both constraints : "
        f"{len(correlated_categories)}\n"
        f"\t{correlated_categories}"
    )
    return categorical_index, correlated_values_index, correlated_categories


def column_distribution(c, y):
    """
    Take a column and return the distribution of the output for each unique value in the column.
    Args:
        c: data vector of a specific feature.
        y: healthy/ill vector.

    Returns:
        dict: mapping dictionary from value to distribution.
    """
    distribution = {}
    for value in np.unique(c):
        value_indexes = np.where(c == value)[0]
        ill = (y[value_indexes] == 1).sum()
        distribution[value] = ill / len(value_indexes)
    return distribution


def distribution(x, y):
    """
    Compute the distribution to every column/feature in the dataset.
    Args:
        x: data array.
        y: healthy/ill vector.
    Returns:
        list[dict]: list of dictionary for all features mapping value to distribution.
    """
    distr = []
    for feature in range(x.shape[1]):
        column = x[:, feature]
        c_distr = column_distribution(column, y)
        distr.append(c_distr)
    return distr


def data_to_distribution(x, map):
    """
    Transform a dataset from categorical values to distribution.

    Args:
        x: data set with categorical values.
        map: list of dictionaries with mapping from values to distribution.

    Returns:
        np.ndarray: numpy array of data mapped from value to distribution.
    """
    mapped_data = np.empty(x.shape, dtype=np.float64)
    for attribute in range(x.shape[1]):
        c = x[:, attribute]
        dictionary = map[attribute]
        c_mapped = np.array([dictionary.get(value, 0) for value in c])
        mapped_data[:, attribute] = c_mapped
    return mapped_data
