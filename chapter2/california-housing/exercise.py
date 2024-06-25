import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import (StratifiedShuffleSplit, cross_val_score,
                                     GridSearchCV, RandomizedSearchCV)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from scipy.stats import randint


rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

def indices_of_top(arr , k):
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k

    def fit(self, X, y=None):
        self.feature_indices_ = indices_of_top(self.feature_importances, self.k)
        return self  # nothing else to do

    def transform(self, X):         
        return X[:]


housing = pd.read_csv('housing.csv')
housing['income_cat'] = pd.cut(housing['median_income'],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_set, test_set = None, None

for train_index, test_index in split.split(housing, housing['income_cat']):
    train_set = housing.loc[train_index]
    test_set = housing.loc[test_index]
assert train_set is not None
assert test_set is not None

for set_ in (train_set, test_set):
    set_.drop('income_cat', axis=1, inplace=True)

train_labels = train_set['median_house_value'].copy()
train_set.drop('median_house_value', axis=1, inplace=True)
test_labels = test_set['median_house_value'].copy()
test_set.drop('median_house_value', axis=1, inplace=True)

train_set_num = train_set.drop('ocean_proximity', axis=1)
num_attributes = list(train_set_num)
cat_attributes = ['ocean_proximity']

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
    ])
full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attributes),
    ('cat', OneHotEncoder(), cat_attributes)
    ])
train_set_prepared = full_pipeline.fit_transform(train_set)


svr = SVR()
svr.fit(train_set_prepared, train_labels)

test_predictions = svr.predict(train_set_prepared)
svr_mse = mean_squared_error(train_labels, test_predictions)
svr_rmse = np.sqrt(svr_mse)
svr_scores = cross_val_score(svr, train_set_prepared, train_labels,
                             scoring='neg_mean_squared_error', cv=10)
svr_rmse_scores = np.sqrt(-svr_scores)

print(svr_rmse)
print(svr_rmse_scores.mean())
print(svr_rmse_scores.std())


def print_search_result(search):
    print('best_params:', search.best_params_)
    print('best_estiamtor:', search.best_estimator_)

    search_results = search.cv_results_
    for mean_score, params in zip(search_results['mean_test_score'],
            search_results['params']):
        print(np.sqrt(-mean_score), params)

# param_grid =  [
#     {'kernel': ['linear'], 'C': [1, 10, 20]},
#     {'kernel': ['rbf'], 'C': [1, 10, 20], 'gamma': ['scale', 'auto', 1, 10, 20]}
#     ]
param_grid = [
    {'kernel': ['linear'], 'C': [20, 40, 60]}
    ]

grid_search = GridSearchCV(svr, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(train_set_prepared, train_labels)

print_search_result(grid_search)


random_param_grid = {
    'kernel': ['linear', 'rbf'],
    'C': randint(1, 100),
    'gamma': randint(0, 10),
    }
randomized_search = RandomizedSearchCV(svr, random_param_grid, cv=5, n_iter=1,
                                       scoring='neg_mean_squared_error',
                                       return_train_score=True)
randomized_search.fit(train_set_prepared, train_labels)
print_search_result(randomized_search)


