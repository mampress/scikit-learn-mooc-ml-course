from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

housing = fetch_california_housing(as_frame=True)
data, target = housing.data, housing.target

target = target*100 # to have house prices in thousands of dollars and not hundred thousand of dollars

# using a tree decision regressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(data, target)

# evaluate the model via mean absolute error, sum(abs(y - y_hat))/n
target_predicted = regressor.predict(data)
score = mean_absolute_error(target, target_predicted)
print(f"On average, our regressor makes an error of {score:.2f} k$") # getting 0 training error, impossible
# comes from the fact that we predicted on the same dataset we predicted on, using 
# decison tree all values are stored in the leaf nodes

# split dataset into training and test set (we want to minimize the error inside the generalized data, not the training data)
data_train, data_test, target_train, target_test = train_test_split(
    data, target, random_state=0
)

regressor.fit(data_train, target_train)

target_predicted = regressor.predict(data_train)
score = mean_absolute_error(target_train, target_predicted)
print(f"The training error of our model is {score:.2f} k$")

target_predicted = regressor.predict(data_test)
score = mean_absolute_error(target_test, target_predicted)
print(f"The testing error of our model is {score:.2f} k$")

from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit

cv = ShuffleSplit(n_splits=40, test_size=0.3, random_state=0)
cv_results = cross_validate(
    regressor, data, target, cv=cv, scoring="neg_mean_absolute_error"
)

import pandas as pd

cv_results = pd.DataFrame(cv_results)

cv_results["test_error"] = -cv_results["test_score"]

print(
    "The mean cross-validated testing error is: "
    f"{cv_results['test_error'].mean():.2f} k$"
)

print(
    "The standard deviation of the testing error is: "
    f"{cv_results['test_error'].std():.2f} k$"
)

# 46.36 ± 1.17 k$.