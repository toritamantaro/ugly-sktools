import functools

import optuna
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split

from util.optuna_tool import OptunaObjectiveFactory


def light_gbm_example():
    from lightgbm import LGBMClassifier
    seed = 0

    ''' light_gbmの最適化パラメータを指定する方法 '''

    # デフォルト（固定）のパラメータを用意しておく
    default_params = {
        # 'objective': 'binary',
        # 'metric': 'auc',
        'verbosity': -1,
        'boosting': 'gbdt',
        'learning_rate': 0.02,
        'n_estimators': 10000,
        'seed': seed,
    }

    '''
    探索するパラメータのキーや探索範囲を以下のような辞書型で指定します。
    'logunirorm'や'int'といったキーは、下記HPの'suggest_loguniform(name, low, high)'や
    'suggest_int(name, low, high[, step, log])'に対応していいます。
    https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial
    '''
    # 探索するパラメータやその範囲の指定
    search_params = {
        'loguniform': {
            'lambda_l1': {'low': 1e-8, 'high': 1e1, },
            'lambda_l2': {'low': 1e-8, 'high': 1e1, },
        },
        'uniform': {
            'subsample': {'low': 0.1, 'high': 0.5, },
        },
        'int': {
            'max_depth': {'low': 4, 'high': 16, },
        },
    }

    # optuna用のobjective関数を生成するクラス
    # func_generator = OptunaObjectiveFactory(scoring='roc_auc')  # <- 最適化の評価指標をaucにしたい場合
    func_generator = OptunaObjectiveFactory(scoring='accuracy')
    # 最適化の対象となるestimatorを用意（この時点で固定パラメータを渡しておく）
    estimator = LGBMClassifier(**default_params)
    # OptunaObjectiveFactory.create()にestimatorとsearch_paramsを渡して実行し、objective関数を生成
    optfunc = func_generator.create(estimator, search_params)

    # 評価用のデータを準備
    X_train, X_test, y_train, y_test = get_wine_data(seed)

    # optunaで最適パラメータ探索
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler(seed=seed))
    study.optimize(functools.partial(optfunc, X=X_train, y=y_train), n_trials=5)

    print('Best trial:', study.best_trial.params)
    print('Best value: ', study.best_trial.value)


def svm_example():
    from sklearn.svm import SVC
    seed = 0

    # 評価用のデータを準備
    X_train, X_test, y_train, y_test = get_wine_data(seed)

    ''' チューニング前の精度確認 '''
    default_params = {
        'random_state': seed,
        'gamma': 'auto',
        'kernel': 'rbf'
    }

    m = SVC(**default_params)
    # kFold
    scores = cross_validate(m, X_train, y_train, cv=10, n_jobs=-1, scoring='accuracy')
    print("kFold accuracy of SVM : {:.3f}".format(scores['test_score'].mean()))

    ''' optunaによる最適化 '''
    # 自作クラス(OptunaObjectiveFactory)を用いて、optunaのobjective関数を生成する（SVM用）
    search_params = {
        'float': {
            'C': {'low': 1e-10, 'high': 1e10, 'log': True, },
        },
    }

    func_generator = OptunaObjectiveFactory(scoring='accuracy', n_folds=10)
    optfunc = func_generator.create(SVC(**default_params), search_params)

    # optunaによる最適化学習
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler(seed=seed))
    study.optimize(functools.partial(optfunc, X=X_train, y=y_train), n_trials=10)

    print('Best trial:', study.best_trial.params)
    print('Best value: ', study.best_trial.value)

    ''' チューニング後の精度確認 '''
    best_params = study.best_trial.params
    best_params.update(default_params)
    print(best_params)
    m = SVC(**best_params)
    # kFold
    scores = cross_validate(m, X_train, y_train, cv=10, n_jobs=-1, scoring='accuracy')
    print("kFold accuracy of SVM : {:.3f}".format(scores['test_score'].mean()))


def get_wine_data(seed: int = 0):
    # データの読み込み
    wine = load_wine()
    X = wine['data']
    y = wine['target']

    ''' データの分割と標準化の実施 '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    # #標準化
    # sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    light_gbm_example()
    # svm_example()
