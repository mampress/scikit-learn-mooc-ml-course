import pandas as pd

from sklearn.model_selection import train_test_split

adult_census = pd.read_csv("./adult_census.csv")

target_name = "workclass"
target = adult_census[target_name]
data = adult_census.drop(columns=[target_name, "education-num"])

data_train, data_test, target_train, target_test = train_test_split(
    data, target, train_size=0.2, random_state=42
)

from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OrdinalEncoder

categorical_preprocessor = OrdinalEncoder(
    handle_unknown="use_encoded_value", unknown_value=-1
)
preprocessor = ColumnTransformer(
    [
        (
            "cat_preprocessor",
            categorical_preprocessor,
            selector(dtype_include=object),
        )
    ],
    remainder="passthrough",
)

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

model = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", HistGradientBoostingClassifier(random_state=42)),
    ]
)

from sklearn.model_selection import cross_val_score # just returns less stuff than cross_validate


learning_rates = [0.01, 0.1, 1, 10]
max_leaf_nodes = [3, 10, 30]

best_cross_val_score = -1

for learning_rate in learning_rates:
    for max_leaf_node in max_leaf_nodes:
        model.set_params(classifier__learning_rate=learning_rate)
        model.set_params(classifier__max_leaf_nodes=max_leaf_node)
        scores = cross_val_score(model, data_train, target_train, cv=2) # only on the training data, as test data is used for evaluation
        mean_score = scores.mean()
        if mean_score > best_cross_val_score:
            best_cross_val_score = mean_score
            best_learn_rate = learning_rate
            best_max_leaf_node = max_leaf_node

print(f"Best Accuracy: {best_cross_val_score}, best learning rate {best_learn_rate}, best max_leaf_nodes {best_max_leaf_node}")

model.set_params(classifier__learning_rate=best_learn_rate)
model.set_params(classifier__max_leaf_nodes=best_max_leaf_node)

model.fit(data_train, target_train) # fit the model with training data
test_score = model.score(data_test, target_test) # evaluate the score having also the test target, predict() would be used not having the target

print(f"Test score after the parameter tuning: {test_score:.3f}")