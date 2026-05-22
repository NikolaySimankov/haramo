###########
# Imports #
###########

import itertools
from typing import Union

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.svm import SVC, NuSVC
from sklearn.linear_model import SGDClassifier, LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.pipeline import Pipeline

from ..utils import instantiate_scaler, filter_args
from ..feature_selection import instantiate_feature_selector

from optuna import Trial

#############
# Functions #
#############


def instantiate_RBFSVM_Classifier(
    trial: Trial, hyperparameters: str = "optimize", **kwargs
):
    if hyperparameters == "optimize":
        params = {
            "C": trial.suggest_power("C", -4, 1, base=10),
            "gamma": trial.suggest_power("gamma", -4, 1, base=10),
        }
    elif hyperparameters == "default":
        params = {}
    else:
        raise ValueError("hyperparameters must be 'optimize' or 'default'")
    params.update(filter_args(SVC, **kwargs))
    estimator = SVC(kernel="rbf", **params)
    return estimator


def instantiate_LSVM_Classifier(
    trial: Trial, hyperparameters: str = "optimize", **kwargs
):
    if hyperparameters == "optimize":
        params = {
            "gamma": trial.suggest_power("gamma", -4, 1, base=10),
            "degree": trial.suggest_int("degree", 2, 4),
            "kernel": trial.suggest_categorical("kernel", ["linear", "poly"]),
        }
    elif hyperparameters == "default":
        params = {}
    else:
        raise ValueError("hyperparameters must be 'optimize' or 'default'")
    params.update(filter_args(SVC, **kwargs))
    estimator = SVC(**params)
    return estimator


def instantiate_NuLSVM_Classifier(
    trial: Trial, hyperparameters: str = "optimize", **kwargs
):
    if hyperparameters == "optimize":
        params = {
            "nu": trial.suggest_power("nu", -2, 0, base=10),
            "gamma": trial.suggest_power("gamma", -4, 1, base=10),
            "degree": trial.suggest_int("degree", 2, 4),
            "kernel": trial.suggest_categorical("kernel", ["linear", "poly"]),
        }
    elif hyperparameters == "default":
        params = {}
    else:
        raise ValueError("hyperparameters must be 'optimize' or 'default'")
    params.update(filter_args(NuSVC, **kwargs))
    estimator = NuSVC(**params)
    return estimator


def instantiate_NuRBFSVM_Classifier(
    trial: Trial, hyperparameters: str = "optimize", **kwargs
):
    if hyperparameters == "optimize":
        params = {
            "nu": trial.suggest_power("nu", -4, 1, base=10),
            "gamma": trial.suggest_power("gamma", -4, 1, base=10),
        }
    elif hyperparameters == "default":
        params = {}
    else:
        raise ValueError("hyperparameters must be 'optimize' or 'default'")
    params.update(filter_args(NuSVC, **kwargs))
    estimator = NuSVC(kernel="rbf", **params)
    return estimator


def instantiate_SGD_Classifier(
    trial: Trial, hyperparameters: str = "optimize", **kwargs
):
    if hyperparameters == "optimize":
        params = {
            "alpha": trial.suggest_power("SGD_alpha", -4, 1, base=10),
        }
    elif hyperparameters == "default":
        params = {}
    else:
        raise ValueError("hyperparameters must be 'optimize' or 'default'")
    params.update(filter_args(SGDClassifier, **kwargs))
    estimator = SGDClassifier(**params)
    return estimator


def instantiate_MLP_Classifier(
    trial: Trial, hyperparameters: str = "optimize", **kwargs
):
    if hyperparameters == "optimize":
        params = {
            "activation": trial.suggest_categorical(
                "activation", ["identity", "logistic", "tanh", "relu"]
            ),
            "solver": trial.suggest_categorical("MLP_solver", ["lbfgs", "sgd", "adam"]),
            "hidden_layer_sizes": trial.suggest_categorical(
                "hidden_layer_sizes",
                list(itertools.product([50, 100, 200], repeat=2))
                + list(itertools.product([50, 100, 200], repeat=3)),
            ),
            "alpha": trial.suggest_power("MLP_alpha", -4, 1, base=10),
        }
    elif hyperparameters == "default":
        params = {}
    else:
        raise ValueError("hyperparameters must be 'optimize' or 'default'")
    params.update(filter_args(MLPClassifier, **kwargs))
    estimator = MLPClassifier(early_stopping=True, learning_rate="adaptive", **params)
    return estimator


def instantiate_RF_Classifier(
    trial: Trial, hyperparameters: str = "optimize", **kwargs
):
    if hyperparameters == "optimize":
        params = {
            "n_estimators": trial.suggest_power("RF_n_estimators", 7, 11, base=2),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 5, 35, step=5),
        }
    elif hyperparameters == "default":
        params = {}
    else:
        raise ValueError("hyperparameters must be 'optimize' or 'default'")
    params.update(filter_args(RandomForestClassifier, **kwargs))
    estimator = RandomForestClassifier(
        min_samples_split=6, min_samples_leaf=3, **params
    )
    return estimator


def instantiate_ET_Classifier(
    trial: Trial, hyperparameters: str = "optimize", **kwargs
):
    if hyperparameters == "optimize":
        params = {
            "n_estimators": trial.suggest_power("ET_n_estimators", 7, 11, base=2),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 5, 35, step=5),
        }
    elif hyperparameters == "default":
        params = {}
    else:
        raise ValueError("hyperparameters must be 'optimize' or 'default'")
    params.update(filter_args(ExtraTreesClassifier, **kwargs))
    estimator = ExtraTreesClassifier(min_samples_split=6, min_samples_leaf=3, **params)
    return estimator


def instantiate_LGBM_Classifier(
    trial: Trial, hyperparameters: str = "optimize", **kwargs
):
    if hyperparameters == "optimize":
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 5, 50, step=5),
            "learning_rate": trial.suggest_power("learning_rate", -3, 0, base=10),
            "n_estimators": trial.suggest_power("n_estimators", 7, 11, base=2),
            "reg_alpha": trial.suggest_power("reg_alpha", -3, 1, base=10),
            "reg_lambda": trial.suggest_power("reg_lambda", -3, 1, base=10),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.2, 1.0, step=0.1
            ),
        }
    elif hyperparameters == "default":
        params = {}
    else:
        raise ValueError("hyperparameters must be 'optimize' or 'default'")

    kwargs["verbose"] = -1
    params.update(filter_args(LGBMClassifier, **kwargs))
    model = LGBMClassifier(
        force_col_wise=True, objective="binary", verbosity=-1, **params
    )
    return model


def instantiate_XGB_Classifier(
    trial: Trial, hyperparameters: str = "optimize", **kwargs
):
    if hyperparameters == "optimize":
        params = {
            "max_leaves": trial.suggest_int("max_leaves", 5, 50, step=5),
            "learning_rate": trial.suggest_power("learning_rate", -3, 0, base=10),
            "n_estimators": trial.suggest_power("n_estimators", 7, 11, base=2),
            "reg_alpha": trial.suggest_power("reg_alpha", -3, 1, base=10),
            "reg_lambda": trial.suggest_power("reg_lambda", -3, 1, base=10),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.2, 1.0, step=0.1
            ),
            "booster": trial.suggest_categorical(
                "booster", ["gbtree", "gblinear", "dart"]
            ),
        }
    elif hyperparameters == "default":
        params = {}
    else:
        raise ValueError("hyperparameters must be 'optimize' or 'default'")

    kwargs["verbosity"] = 0
    kwargs["tree_method"] = "approx"
    params.update(filter_args(XGBClassifier, **kwargs))
    model = XGBClassifier(objective="binary:logistic", **params)
    return model


def instantiate_CatBoost_Classifier(
    trial: Trial, hyperparameters: str = "optimize", **kwargs
):
    if hyperparameters == "optimize":
        params = {
            "depth": trial.suggest_int("depth", 3, 10, step=1),
            "learning_rate": trial.suggest_power("learning_rate", -3, 0, base=10),
            "iterations": trial.suggest_power("iterations", 7, 11, base=2),
            "l2_leaf_reg": trial.suggest_power("l2_leaf_reg", -3, 1, base=10),
            "colsample_bylevel": trial.suggest_float(
                "colsample_bylevel", 0.2, 1.0, step=0.1
            ),
        }
    elif hyperparameters == "default":
        params = {}
    else:
        raise ValueError("hyperparameters must be 'optimize' or 'default'")

    kwargs["thread_count"] = kwargs.pop("n_jobs", 1)
    params.update(filter_args(CatBoostClassifier, **kwargs))
    model = CatBoostClassifier(**params)
    return model


def instantiate_KNN_Classifier(
    trial: Trial, hyperparameters: str = "optimize", **kwargs
):
    if hyperparameters == "optimize":
        params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 5, 50, step=5),
            "leaf_size": trial.suggest_int("leaf_size", 10, 50, step=10),
            "p": trial.suggest_int("p", 1, 2),
        }
    elif hyperparameters == "default":
        params = {}
    else:
        raise ValueError("hyperparameters must be 'optimize' or 'default'")
    params.update(filter_args(KNeighborsClassifier, **kwargs))
    estimator = KNeighborsClassifier(weights="distance", **params)
    return estimator


def instantiate_ENet_Classifier(
    trial: Trial, hyperparameters: str = "optimize", **kwargs
):
    if hyperparameters == "optimize":
        params = {
            "C": trial.suggest_power("C", -4, 1, base=10),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0, step=0.1),
        }
    elif hyperparameters == "default":
        params = {}
    else:
        raise ValueError("hyperparameters must be 'optimize' or 'default'")
    params.update(filter_args(LogisticRegression, **kwargs))
    estimator = LogisticRegression(solver="saga", **params)
    return estimator


def instantiate_PrimalLR_Classifier(
    trial: Trial, hyperparameters: str = "optimize", **kwargs
):
    if hyperparameters == "optimize":
        params = {
            "C": trial.suggest_power("C", -4, 1, base=10),
            "solver": trial.suggest_categorical(
                "LR_solver", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
            ),
        }
    elif hyperparameters == "default":
        params = {}
    else:
        raise ValueError("hyperparameters must be 'optimize' or 'default'")
    params.update(filter_args(LogisticRegression, **kwargs))
    estimator = LogisticRegression(penalty="l2", dual=False, **params)
    return estimator


def instantiate_DualLR_Classifier(
    trial: Trial, hyperparameters: str = "optimize", **kwargs
):
    if hyperparameters == "optimize":
        params = {
            "C": trial.suggest_power("C", -4, 1, base=10),
        }
    elif hyperparameters == "default":
        params = {}
    else:
        raise ValueError("hyperparameters must be 'optimize' or 'default'")
    params.update(filter_args(LogisticRegression, **kwargs))
    estimator = LogisticRegression(
        penalty="l2", solver="liblinear", dual=True, **params
    )
    return estimator


def instantiate_Ridge_Classifier(
    trial: Trial, hyperparameters: str = "optimize", **kwargs
):
    if hyperparameters == "optimize":
        params = {
            "alpha": trial.suggest_power("alpha", -4, 1, base=10),
        }
    elif hyperparameters == "default":
        params = {}
    else:
        raise ValueError("hyperparameters must be 'optimize' or 'default'")
    params.update(filter_args(RidgeClassifier, **kwargs))
    estimator = RidgeClassifier(solver="auto", **params)
    return estimator


def instantiate_LDA_Classifier(
    trial: Trial, hyperparameters: str = "optimize", **kwargs
):

    if hyperparameters == "optimize":
        params = {
            "solver": trial.suggest_categorical("LDA_solver", ["svd", "lsqr", "eigen"]),
            "n_components": trial.suggest_int("n_components", 1),
        }
    elif hyperparameters == "default":
        params = {}
    else:
        raise ValueError("hyperparameters must be 'optimize' or 'default'")

    params.update(filter_args(LinearDiscriminantAnalysis, **kwargs))
    estimator = LinearDiscriminantAnalysis(**params)
    return estimator


def instantiate_model(
    trial: Trial,
    random_state: int = 42,
    n_jobs: int = 1,
    algorithm: Union[str, list] = "optimize",
    hyperparameters: str = "optimize",
):
    """
    Instantiate a machine learning model based on the given algorithm and trial.

    Parameters:
    trial (Trial): An Optuna trial object used for hyperparameters optimization.
    random_state (int, optional): Random state for reproducibility. Defaults to 42.
    algorithm (Union[str, list], optional): The algorithm to use or 'Optimize' to select the best algorithm. Defaults to 'Optimize'.
    hyperparameters (str, optional): Whether to optimize hyperparameters. Defaults to 'Optimize'.

    Returns:
    model: An instantiated machine learning model.
    """

    if algorithm == "optimize":
        algorithm = trial.suggest_categorical(
            "algorithm",
            [
                "LSVM",
                "RBFSVM",
                "NuLSVM",
                "NuRBFSVM",
                "SGD",
                "MLP",
                "RF",
                "ET",
                "LGBM",
                "XGB",
                "CatB",
                "KNN",
                "ENet",
                "PLR",
                "DLR",
                "Ridge",
                "LDA",
            ],
        )
    elif isinstance(algorithm, list):
        algorithm = trial.suggest_categorical(
            "algorithm",
            algorithm,
        )

    kwargs = {
        "cache_size": 2**11,
        "tol": 1e-4,
        "class_weight": "balanced",
        "probability": True,
        "verbose": 0,
        "max_iter": 2**13,
        "random_state": random_state,
        "n_jobs": n_jobs,
    }

    if algorithm == "LSVM":
        model = instantiate_LSVM_Classifier(trial, hyperparameters, **kwargs)

    elif algorithm == "RBFSVM":
        model = instantiate_RBFSVM_Classifier(trial, hyperparameters, **kwargs)

    elif algorithm == "NuLSVM":
        model = instantiate_NuLSVM_Classifier(trial, hyperparameters, **kwargs)

    elif algorithm == "NuRBFSVM":
        model = instantiate_NuRBFSVM_Classifier(trial, hyperparameters, **kwargs)

    elif algorithm == "SGD":
        model = instantiate_SGD_Classifier(trial, hyperparameters, **kwargs)

    elif algorithm == "MLP":
        model = instantiate_MLP_Classifier(trial, hyperparameters, **kwargs)

    elif algorithm == "RF":
        model = instantiate_RF_Classifier(trial, hyperparameters, **kwargs)

    elif algorithm == "ET":
        model = instantiate_ET_Classifier(trial, hyperparameters, **kwargs)

    elif algorithm == "LGBM":
        model = instantiate_LGBM_Classifier(trial, hyperparameters, **kwargs)

    elif algorithm == "XGB":
        model = instantiate_XGB_Classifier(trial, hyperparameters, **kwargs)

    elif algorithm == "CatB":
        model = instantiate_CatBoost_Classifier(trial, hyperparameters, **kwargs)

    elif algorithm == "KNN":
        model = instantiate_KNN_Classifier(trial, hyperparameters, **kwargs)

    elif algorithm == "ENet":
        model = instantiate_ENet_Classifier(trial, hyperparameters, **kwargs)

    elif algorithm == "PLR":
        model = instantiate_PrimalLR_Classifier(trial, hyperparameters, **kwargs)

    elif algorithm == "DLR":
        model = instantiate_DualLR_Classifier(trial, hyperparameters, **kwargs)

    elif algorithm == "Ridge":
        model = instantiate_Ridge_Classifier(trial, hyperparameters, **kwargs)

    elif algorithm == "LDA":
        model = instantiate_LDA_Classifier(trial, hyperparameters, **kwargs)

    else:
        raise ValueError(
            f"Unknown algorithm: {algorithm}.Valid algorithms are: 'LSVM', 'RBFSVM', 'NuLSVM', 'NuRBFSVM', 'SGD', 'MLP', 'RF', 'ET', 'LGBM', 'XGB', 'CatB', 'KNN', 'ENet', 'PLR', 'DLR', 'Ridge', 'LDA'."
        )

    return model


def instantiate_pipeline(
    trial: Trial,
    task: str = "classification",
    feature_selector: Union[str, list] = "optimize",
    scaler: Union[str, list] = "optimize",
    algorithm: Union[str, list] = "optimize",
    hyperparameters: str = "optimize",
    random_state: int = 42,
    n_jobs: int = 1,
):

    feature_selector = instantiate_feature_selector(
        trial,
        task=task,
        method=feature_selector,
        hyperparameters=hyperparameters,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    scaler = instantiate_scaler(
        trial,
        method=scaler,
        hyperparameters=hyperparameters,
        random_state=random_state,
    )

    model = instantiate_model(
        trial,
        algorithm=algorithm,
        hyperparameters=hyperparameters,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    pipeline = Pipeline(
        [
            ("feature_selector", feature_selector),
            ("scaler", scaler),
            ("model", model),
        ]
    )

    return pipeline
