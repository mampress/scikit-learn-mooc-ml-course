import pandas as pd

blood_transfusion = pd.read_csv("./blood_transfusion.csv")
target_name = "Class"
data = blood_transfusion.drop(columns=target_name)
target = blood_transfusion[target_name] # binary classification problem

from sklearn.dummy import DummyClassifier

dummy_clf = DummyClassifier(random_state=0)

from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit

cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
cv_results = cross_validate(
    dummy_clf, data, target, cv=cv, scoring="accuracy"
)

print(
    "The mean cross-validated accuracy is: "
    f"{cv_results['test_score'].mean():.2f}" # 75%
)

cv_results = cross_validate(
    dummy_clf, data, target, cv=cv, scoring="balanced_accuracy"
)

print(
    "The mean cross-validated balance_accuracy is: "
    f"{cv_results['test_score'].mean():.2f}" # 50%
)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=10)) # 5 - neighbours by default

cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
cv_results = cross_validate(
    model, data, target, cv=cv, scoring="accuracy", return_train_score=True
)

print(
    "The mean cross-validated test accuracy is: "
    f"{cv_results['test_score'].mean():.2f}" # 79%
)
print(
    "The mean cross-validated train accuracy is: "
    f"{cv_results['train_score'].mean():.2f}" # 81%
)

# looking at those scores the model seem to generalize well, as the scores are close to one another


import numpy as np
param_range = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500])

results = dict()

for k in param_range:
    model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    cv_results = cross_validate(
        model, data, target, cv=cv, scoring="balanced_accuracy", return_train_score=True
    )

    results[k] = (cv_results['test_score'].mean(), cv_results['train_score'].mean())

print(results)

"""
n_neighbours  (test_score, train_score)
1: 0.589, 0.893 underfit
2: 0.610, 0.843 underfit
5: 0.609, 0.68.5
10: 0.654, 0.677 generalize
20: 0.640, 0.662 generalize
50: 0.587, 0.591 generalize
100: 0.51, 0.51
200: 0.5, 0.5 overfit
500: 0.5, 0.5 overfit
"""