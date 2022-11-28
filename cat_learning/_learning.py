from typing import Tuple, Any

from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
import matplotlib.pyplot as plt

from.. import io, view


def catboost_one(x_iloc: int, y_col: str) -> Tuple[CatBoostClassifier, Any]:
    """
    Act one set of catboostClassifier
    ref: https://catboost.ai/en/docs/concepts/python-quickstart

    Parameters
    ----------
    x_iloc : int
        Parameter which use to io.divide_xy
    y_col
        Parameter which use to io.divide_xy

    Returns
    -------
    model : catboost.core.CatBoost
        fitted model
    predict_proba : numpy.ndarray
        predicted probability of each class

    """

    x_df, y_array, feature_cols = io.divide_xy(io.read_data(), x_iloc, y_col)

    train_data, eval_data, train_label, eval_label = train_test_split(x_df, y_array, random_state=0)
    train_dataset = Pool(train_data, train_label, cat_features=feature_cols)
    eval_dataset = Pool(eval_data, eval_label, cat_features=feature_cols)

    model: CatBoostClassifier = CatBoostClassifier(loss_function='MultiClass',
                                                   iterations=1000,
                                                   random_seed=42,
                                                   task_type='GPU',
                                                   learning_rate=0.15)
    print('Start model fitting')
    model.fit(train_dataset,
              eval_set=eval_dataset,
              early_stopping_rounds=100,
              use_best_model=True,
              verbose=True)

    view.cat_plot(model, metric_key='MultiClass')
    plt.title(y_col)
    plt.show()
    view.cat_plot(model, metric_key='MultiClass', y_lim=(0, 0.5))
    plt.title(f'{y_col}_focus')
    plt.show()

    # Get predicted probabilities for each class
    preds_proba = model.predict_proba(eval_dataset)

    return model, preds_proba


