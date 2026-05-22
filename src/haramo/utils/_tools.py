###########
# Imports #
###########
import numpy as np
import pandas as pd

from typing import Union
import inspect

from sklearn.base import BaseEstimator, TransformerMixin

from boruta import BorutaPy
from greedy_boruta import GreedyBorutaPy

from ._evaluation import pearson_scorer

from pandas.api.types import is_bool_dtype, is_numeric_dtype

from itertools import chain

from optuna import logging

import warnings

#############
# Functions #
#############


class BorutaPyWrapper(BorutaPy):

    def fit(self, X, y):
        self._is_dataframe = isinstance(X, pd.DataFrame)
        return super().fit(X, y)

    def transform(self, X, weak=False):
        return self._transform(X, weak)

    def fit_transform(self, X, y, weak=False):
        self.fit(X, y)
        return self._transform(X, weak)

    def _transform(self, X, weak=False):
        # sanity check
        try:
            self.ranking_
        except AttributeError:
            raise ValueError("You need to call the fit(X, y) method first.")

        if weak:
            indices = self.support_ + self.support_weak_
        else:
            indices = self.support_

        if self._is_dataframe:
            X = X.iloc[:, indices]
        else:
            X = X[:, indices]

        return X


class GreedyBorutaPyWrapper(GreedyBorutaPy):

    def fit(self, X, y):
        self._is_dataframe = isinstance(X, pd.DataFrame)
        return super().fit(X, y)

    def transform(self, X, weak=False):
        return self._transform(X, weak)

    def fit_transform(self, X, y, weak=False):
        self.fit(X, y)
        return self._transform(X, weak)

    def _transform(self, X, weak=False):
        try:
            self.ranking_
        except AttributeError:
            raise ValueError("You need to call the fit(X, y) method first.")

        if weak:
            indices = self.support_ + self.support_weak_
        else:
            indices = self.support_

        if self._is_dataframe:
            X = X.iloc[:, indices]
        else:
            X = X[:, indices]

        return X


class TransformerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, scaler: callable):
        self.scaler = scaler

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y=None):
        self.scaler.fit(X, y)
        self.is_fitted_ = True
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]):
        transformed = self.scaler.transform(X)
        if isinstance(X, pd.DataFrame):
            try:
                columns = self.scaler.get_feature_names_out()
            except AttributeError:
                columns = X.columns
            return pd.DataFrame(transformed, columns=columns, index=X.index)
        elif isinstance(X, np.ndarray):
            return transformed
        else:
            raise ValueError("Input must be a DataFrame or ndarray")

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame], y=None):
        self.fit(X, y)
        return self.transform(X)


class PValueFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Custom transformer to filter features based on a correlation metric's p-value.
    """

    def __init__(self, threshold=0.05, correlation_func: callable = pearson_scorer):
        self.threshold = threshold
        self.correlation_func = correlation_func

    def fit(self, X, y):
        """
        Fit the transformer to the data.

        Parameters:
        -----------
        X : Union[np.ndarray, pd.DataFrame]
            Feature matrix.
        y : Union[np.ndarray, pd.Series]
            Target vector.

        Returns:
        --------
        self : object
            Returns self.
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        # Perform correlation
        _, p_value = self.correlation_func(X, y)

        # Select features where p-value is less than the threshold
        self.significant_features_ = X.columns[p_value < self.threshold].tolist()

        return self

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit to data, then transform it.

        Parameters:
        -----------
        X : Union[np.ndarray, pd.DataFrame]
            Feature matrix.
        y : Union[np.ndarray, pd.Series], optional
            Target vector.

        Returns:
        --------
        X_new : Union[np.ndarray, pd.DataFrame]
            Transformed feature matrix with selected features.
        """
        return self.fit(X, y, **fit_params).transform(X)

    def transform(self, X):
        """
        Transform the data by selecting significant features.

        Parameters:
        -----------
        X : Union[np.ndarray, pd.DataFrame]
            Feature matrix.

        Returns:
        --------
        X_new : Union[np.ndarray, pd.DataFrame]
            Transformed feature matrix with selected features.
        """

        X_new = X[self.significant_features_]

        return X_new


def set_verbosity(verbose: int):
    """
    Set the verbosity level for logging.

    Parameters:
    verbose (int): The verbosity level. Set to 0 for critical logging only, or any other value for info logging.

    Returns:
    None
    """

    # Ignore all warnings
    warnings.filterwarnings("ignore")

    # Set logging verbosity
    if verbose == 0:
        # If verbosity level is 0, set logging to show only critical messages
        logging.set_verbosity(logging.CRITICAL)
    else:
        # If verbosity level is greater than 0, set logging to show informational messages
        logging.set_verbosity(logging.INFO)


def detect_dtype(x: Union[np.ndarray, pd.Series, list]):
    """
    Detect the type of a x: continuous, discrete, interval, ratio, binary, or nominal.
    Ensure binary variables are in the format 0 or 1.
    """
    if isinstance(x, list):
        x = pd.Series(x)
    elif isinstance(x, np.ndarray):
        x = pd.Series(x)

    if is_bool_dtype(x):
        x = x.astype(int)
        return "binary"

    elif isinstance(x.dtype, pd.CategoricalDtype):
        unique_values = x.dropna().unique()
        if len(unique_values) == 2:
            return "binary"
        else:
            return "nominal"

    elif is_numeric_dtype(x):
        unique_values = x.dropna().unique()
        if len(unique_values) == 2:
            return "binary"
        else:  # Arbitrary threshold for discrete vs continuous
            differences = np.diff(np.sort(unique_values))
            if np.all(differences == differences[0]):
                return "interval/ordinal"
            elif (x >= 0).all() and (x / x.min()).dropna().apply(
                float.is_integer
            ).all():
                return "ratio"
            else:
                return "continuous"


def union_lists(*lists):
    """
    Returns the union of multiple lists, removing duplicates and sorting the result.

    Parameters:
    *lists: Variable number of lists to be combined.

    Returns:
    A sorted list containing all unique elements from the input lists.
    """

    # use the chain() function from itertools to concatenate all the lists
    concatenated_list = list(chain(*lists))

    # use the set() function to remove duplicates and convert the concatenated list to a set
    unique_set = set(concatenated_list)

    # convert the set back to a list
    final_union = list(unique_set)

    # sort the list and return it as the final union
    final_union.sort()

    return final_union


def filter_args(func: callable, **kwargs):
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def log_int(x, base: int = 2):
    return np.floor(np.log(x) / np.log(base)).astype(int)


def pruner_sampling(y, base: int, n_rungs: int):

    data_size = len(y)
    data_scale = log_int(data_size, base)
    min_scale = data_scale - n_rungs

    return [*map(lambda scale: base**scale, range(min_scale, data_scale + 1))]
