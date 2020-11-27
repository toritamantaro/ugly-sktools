from abc import ABCMeta, abstractmethod
from typing import Callable

import optuna
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate


class FunctionFactory(metaclass=ABCMeta):
    def create(self, estimator: BaseEstimator, search_params: dir) -> Callable[[], float]:
        f = self.create_function(estimator, search_params)
        return f

    @abstractmethod
    def create_function(self, estimator: BaseEstimator, search_params: dir) -> Callable[[], float]:
        pass


class OptunaObjectiveFactory(FunctionFactory):
    """
    sklearn.model_selection.cross_validate()で評価した値を用いて最適化を行う
    Optuna用のobjective関数を生成するクラスです
    """

    def __init__(self, scoring: str = 'roc_auc', n_folds: int = 10):
        """
        Parameters
        ----------
        以下はどちらもsklearn.model_selection.cross_validate()に与える引数になります。
        scoring: str
            cross_validate()の評価方法を指定
        n_folds: int
            cross_validate()の分割数を指定
        """
        self._scoring = scoring  # 'accuracy','roc_auc', etc
        self._n_folds = n_folds

    def create_function(self, estimator: BaseEstimator, search_params: dir) -> Callable[[], float]:
        params = {}

        def objective(trial: optuna.trial.Trial, X, y):
            """
            See the following address for 'suggest_keys'.
            https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial
            """
            suggest_keys = {
                'float': trial.suggest_float,  # (name, low, high, *[, step, log])
                'int': trial.suggest_int,  # (name, low, high[, step, log])
                'categorical': trial.suggest_categorical,  # (name, choices)
                'discrete_uniform': trial.suggest_discrete_uniform,  # (name, low, high, q)
                'loguniform': trial.suggest_loguniform,  # (name, low, high)
                'uniform': trial.suggest_uniform,  # (name, low, high)
            }

            for tk, tv in search_params.items():
                sg = suggest_keys.get(tk, None)
                if sg is not None:
                    for k, v in tv.items():
                        params[k] = sg(k, **v)

            estimator.set_params(**params)

            score = cross_validate(estimator, X, y, n_jobs=-1, cv=self._n_folds, scoring=self._scoring)
            accuracy = score['test_score'].mean()
            return accuracy

        return objective
