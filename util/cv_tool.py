from abc import ABCMeta, abstractmethod
from typing import Optional, Callable, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import BaseCrossValidator


class CvToolBuilder(metaclass=ABCMeta):
    """
    Builderパターンの抽象クラス
    交差検証を行うために必要な一連の処理を統合したもの
    """

    @staticmethod
    def input2pd(
            x_train: Union[np.ndarray, pd.DataFrame],
            y_train: Union[np.ndarray, pd.Series],
            x_test: Optional[Union[np.ndarray, pd.DataFrame]],
    ):
        x_train = pd.DataFrame(x_train)
        y_train = pd.Series(y_train)
        x_test = None if x_test is None else pd.DataFrame(x_test)
        return x_train, y_train, x_test

    @staticmethod
    def create_estimators(
            estimator: BaseEstimator,
            n_splits: int
    ):
        # 交差検証用のモデルを交差分割数ぶん生成
        params = estimator.get_params()
        estimators = [estimator.__class__(**params) for _ in range(n_splits)]
        return estimators

    def create_stack_yard(
            self,
            x_train_len: int,
            x_test_len: int,
            y_unique_num: int,
            n_splits: int
    ):
        # 結果を格納する入れ物を生成
        oof_shape, test_pred_shape = self.make_stack_shapes(x_train_len, x_test_len, y_unique_num)
        oof_stk = np.empty(oof_shape)
        oof_scores_stk = np.empty(n_splits)
        test_pred_stk = None if x_test_len is None else np.empty((n_splits, *test_pred_shape))
        return oof_stk, oof_scores_stk, test_pred_stk

    def out_of_fold(
            self,
            estimator: BaseEstimator,
            train_x, train_y,
            valid_x, valid_y):
        # lightGBMとcatboostの場合は、fit時に下記パラメータを与える
        fit_params = {}
        if type(estimator).__name__ in ('LGBMClassifier', 'CatBoostClassifier',):
            if 'eval_set' not in fit_params:
                fit_params['eval_set'] = [(valid_x, valid_y)]
            if 'early_stopping_rounds' not in fit_params:
                fit_params['early_stopping_rounds'] = 100

        estimator.fit(train_x, train_y, **fit_params)
        oof = self.make_pred(estimator, valid_x)
        return oof

    def oof_score(self, out_of_fold_stk, valid_idx, valid_y, score_func):
        pred_y = self.make_oof_score_target(out_of_fold_stk, valid_idx)
        oof_score = score_func(valid_y, pred_y)
        return oof_score

    @abstractmethod
    def make_stack_shapes(self, x_train_len, x_test_len, y_unique_num):
        pass

    @abstractmethod
    def make_oof_score_target(self, oof_stk, valid_idx):
        pass

    @abstractmethod
    def make_pred(self, estimator, x):
        pass


class PredictProbaBuilder(CvToolBuilder):
    """
    具体的なBuilderクラス
    交差検証を行いながらpredict_probaを実施するために必要な具体的な処理を定義
    """

    def make_stack_shapes(self, x_train_len, x_test_len, y_unique_num):
        # 結果を格納する入れ物のシェイプを生成
        oof_shape = (x_train_len, y_unique_num)
        test_pred_shape = (x_test_len, y_unique_num)
        return oof_shape, test_pred_shape

    def make_oof_score_target(self, oof_stk, valid_idx):
        target_idx = 1
        return oof_stk[valid_idx][:, target_idx]

    def make_pred(self, estimator, x):
        return estimator.predict_proba(x)


class PredictBuilder(CvToolBuilder):
    """
    具体的なBuilderクラス
    交差検証を行いながらpredictを実施するために必要な具体的な処理を定義
    """

    def make_stack_shapes(self, x_train_len, x_test_len, y_unique_num):
        # 結果を格納する入れ物のシェイプを生成
        oof_shape = (x_train_len,)
        test_pred_shape = (x_test_len,)
        return oof_shape, test_pred_shape

    def make_oof_score_target(self, oof_stk, valid_idx):
        return oof_stk[valid_idx]

    def make_pred(self, estimator, x):
        return estimator.predict(x)


class CvToolDirector(object):
    """
    CvToolBuilderクラスの取り纏め役
    Builderクラスで生成した処理を取りまとめて、交差検証を実行する
    """

    def __init__(
            self,
            builder: CvToolBuilder,
            estimator: BaseEstimator,  # BaseEstimatorのインスタンを渡す
            splitter: BaseCrossValidator,  # BaseCrossVarlidatorのインスタンを渡す
            score_func: Optional[Callable[[], float]] = roc_auc_score,  # scoreの計算に用いる関数を渡す
    ):
        self._builder = builder
        self._estimator = estimator
        self._splitter = splitter
        self._score_func = score_func

    def cv_pred(
            self,
            x_train: Union[np.ndarray, pd.DataFrame],
            y_train: Union[np.ndarray, pd.Series],
            x_test: Optional[Union[np.ndarray, pd.DataFrame]],
    ):
        # 入力データをDataFrameやSeriesに変換する
        x_train, y_train, x_test = self._builder.input2pd(x_train, y_train, x_test)

        # 処理に必要な値を確認
        n_splits = self._splitter.get_n_splits()
        y_unique_num = y_train.nunique(dropna=True)
        x_train_len = len(x_train)
        x_test_len = len(x_test)

        # 交差検証用のモデルを交差分割数ぶん生成
        estimators = self._builder.create_estimators(self._estimator, n_splits)

        # 結果を格納する入れ物を生成
        oof_stk, oof_scores_stk, test_pred_stk = self._builder.create_stack_yard(
            x_train_len,
            x_test_len,
            y_unique_num,
            n_splits,
        )

        # 交差検証
        for i, (train_idx, valid_idx) in enumerate(self._splitter.split(x_train, y_train)):
            train_x, train_y = x_train.iloc[train_idx], y_train.iloc[train_idx]
            valid_x, valid_y = x_train.iloc[valid_idx], y_train.iloc[valid_idx]

            # 交差検証による学習
            oof_stk[valid_idx] = self._builder.out_of_fold(
                estimators[i],
                train_x, train_y,
                valid_x, valid_y,
            )

            # スコアの計算
            if self._score_func is not None:
                oof_scores_stk[i] = self._builder.oof_score(
                    oof_stk,
                    valid_idx,
                    valid_y,
                    self._score_func,
                )
                print("Fold {0} score : {1:.6f}".format(i + 1, oof_scores_stk[i]))

            # testデータの予測
            if x_test is not None:
                test_pred_stk[i, :] = self._builder.make_pred(estimators[i], x_test)

        return oof_stk, oof_scores_stk, test_pred_stk


class CvToolClient(object):
    """
    CvToolDirectorクラスのClient
    処理したい内容によって、生成するCvToolBuilderクラスを切り替えたり、
    CvToolDirectorクラスが生成した結果を受けて、その平均値やモードなどを計算する
    """

    def __init__(
            self,
            estimator: BaseEstimator,  # BaseEstimatorのインスタンを渡す
            splitter: BaseCrossValidator,  # BaseCrossVarlidatorのインスタンを渡す
            score_func: Optional[Callable[[], float]] = roc_auc_score,  # scoreの計算に用いる関数を渡す
    ):
        self._estimator = estimator
        self._splitter = splitter
        self._score_func = score_func

    def predict_proba(self, x_train, y_train, x_test=None, vote='mean'):
        builder = PredictProbaBuilder()
        oof, oof_score, pred = self.cv_voted_pred(x_train, y_train, x_test, builder, vote)
        return oof, oof_score, pred

    def predict(self, x_train, y_train, x_test=None, vote='mode'):
        builder = PredictBuilder()
        oof, oof_score, pred = self.cv_voted_pred(x_train, y_train, x_test, builder, vote)
        return oof, oof_score, pred

    def cv_voted_pred(
            self,
            x_train: Union[np.ndarray, pd.DataFrame],
            y_train: Union[np.ndarray, pd.Series],
            x_test: Optional[Union[np.ndarray, pd.DataFrame]],
            builder: CvToolBuilder,
            vote: str,
    ):
        """
        estimatorで予測した結果の平均値もしくは最頻値を計算して返す
        """
        director = CvToolDirector(
            builder,
            self._estimator,
            self._splitter,
            self._score_func
        )

        oof_stk, oof_scores_stk, test_pred_stk = director.cv_pred(x_train, y_train, x_test)

        oof = oof_stk
        oof_score = oof_scores_stk.mean()

        vote_method = {
            'mean': self.predict_stack_mean,
            'mode': self.predict_stack_mode,
        }

        pred = None
        if x_test is not None:
            pred = vote_method.get(vote, self.predict_stack_mean)(test_pred_stk)

        return oof, oof_score, pred

    def predict_stack_mode(self, arr: np.ndarray):
        vote = np.apply_along_axis(self.minvalue_of_mode, 0, arr)
        return vote

    @staticmethod
    def predict_stack_mean(arr: np.ndarray):
        vote = arr.mean(axis=0)
        return vote

    @staticmethod
    def minvalue_of_mode(arr: np.ndarray):
        """
        複数の最頻値のうち最小の値のみを返す
        """
        uniqs, counts = np.unique(arr, return_counts=True)
        mode = uniqs[counts == np.amax(counts)].min()
        return mode
