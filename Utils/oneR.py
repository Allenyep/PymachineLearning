
#实现OneR算法
#思路：根据已有数据中，具有相同特征值的个体最可能属于哪个类别进行分类
from sklearn.datasets import load_iris
import numpy as np

dataset = load_iris()
X = dataset.data
y = dataset.target
n_samples, n_features = X.shape

attribute_means = X.mean(axis=0)
assert attribute_means.shape == (n_features,)
X_d = np.array(X >= attribute_means, dtype='int')

from sklearn.cross_validation import train_test_split

Xd_train, Xd_test, y_train, y_test = train_test_split(X_d, y, random_state=14)
print("There are {} training samples".format(y_train.shape))
print("There are {} testing samples".format(y_test.shape))

from collections import defaultdict
from operator import itemgetter


def train(X, y_true, feature):
    n_samples, n_features = X.shape
    assert 0 <= feature < n_features
    # Get all of the unique values that this variable has
    values = set(X[:, feature])
    # Stores the predictors array that is returned
    predictors = dict()
    errors = []
    for current_value in values:
        most_frequent_class, error = train_feature_value(X, y_true, feature, current_value)
        predictors[current_value] = most_frequent_class
        errors.append(error)
    # Compute the total error of using this feature to classify on
    total_error = sum(errors)
    return predictors, total_error


def train_feature_value2(X, y_true, feature_index, value):
    class_counts = defaultdict(int)
    for sample, y in zip(X, y_true):
        if sample[feature_index] == value:
            class_counts[y] += 1
    sorted_class_counts = sorted(class_counts.items(), key=itemgetter(1), reverse=True)
    most_frequent_class = sorted_class_counts[0][0]
    incorrect_predictions = [class_count for class_value, class_count in class_counts.items() if
                             class_value != most_frequent_class]
    error = sum(incorrect_predictions)
    return most_frequent_class, error


def train_feature_value(X, y_true, feature, value):
    # Create a simple dictionary to count how frequency they give certain predictions
    class_counts = defaultdict(int)
    # Iterate through each sample and count the frequency of each class/value pair
    for sample, y in zip(X, y_true):
        if sample[feature] == value:
            class_counts[y] += 1
    # Now get the best one by sorting (highest first) and choosing the first item
    sorted_class_counts = sorted(class_counts.items(), key=itemgetter(1), reverse=True)
    most_frequent_class = sorted_class_counts[0][0]
    # The error is the number of samples that do not classify as the most frequent class
    # *and* have the feature value.
    n_samples = X.shape[1]
    error = sum([class_count for class_value, class_count in class_counts.items()
                 if class_value != most_frequent_class])
    return most_frequent_class, error


def train_on_feature(X, y_true, feature_index):
    values = set(X[:, feature_index])
    predictors = {}
    errors = []
    for current_value in values:
        most_frequent_class, error = train_feature_value(X, y_true, feature_index, current_value)
        predictors[current_value] = most_frequent_class
        errors.append(error)
    total_error = sum(errors)
    return predictors, total_error


all_predictors = {variable: train(Xd_train, y_train, variable) for variable in range(Xd_train.shape[1])}
errors = {variable: error for variable, (mapping, error) in all_predictors.items()}
# for feature_index in range(Xd_train.shape[1]):
#     predictors, total_error = train_on_feature(Xd_train, y_train, feature_index)
#     all_predictors[feature_index] = predictors
#     errors[feature_index] = total_error
best_variable, best_error = sorted(errors.items(), key=itemgetter(1))[0]
print("The best model is based on variable {0} and has error {1:.2f}".format(best_variable, best_error))
model = {'variable': best_variable,
         'predictor': all_predictors[best_variable][0]}
print(model)


def predict(X_test, model):
    variable = model['variable']
    predictor = model['predictor']
    y_predicted = np.array([predictor[int(sample[variable])] for sample in X_test])
    return y_predicted


y_predicted = predict(Xd_test, model)
print(y_predicted)

accuracy = np.mean(y_predicted == y_test) * 100

print("the test accuracy is {:.1f}%".format(accuracy))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_predicted))