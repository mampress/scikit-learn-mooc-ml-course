from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

data, target = fetch_california_housing(return_X_y=True, as_frame=True)
target *= 100  # rescale the target in k$

data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=42
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

model = Pipeline([
    ("preprocessor", StandardScaler()), 
    ("knn", KNeighborsRegressor())
])

from sklearn.model_selection import RandomizedSearchCV

param_distributions = {
    "preprocessor__with_mean" : [True, False],
    "preprocessor__with_std" : [True, False],
    "knn__n_neighbors" : np.logspace(0, 3, num=10).astype(np.int32)
}

model_random_search = RandomizedSearchCV(model, param_distributions=param_distributions, n_iter=20, cv=5, verbose=1)
model_random_search.fit(data_train, target_train)

from pprint import pprint
print("The best parameter are: ")
pprint(model_random_search.best_params_)