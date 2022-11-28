import matplotlib.pyplot as plt


def cat_plot(model, metric_key: str = 'Logloss', y_lim: tuple = (0, 3)):
    """
    Show plot of train and eval metric of catboost model
    ref: https://blog.amedama.jp/entry/catboost

    Parameters
    ----------
    model : catboost.core.CatBoost
        fit was finished catboost model
    metric_key : str, optional
        key of loss_function which selected in model.fit(), default is Logloss
    y_lim : tuple, optional
        Tuple of min and max value for plot

    Returns
    -------
    None

    """

    # メトリックの推移を取得する
    history = model.get_evals_result()
    # グラフにプロットする
    train_metric = history['learn'][metric_key]
    plt.plot(train_metric, label='train metric')
    eval_metric = history['validation'][metric_key]
    plt.plot(eval_metric, label='eval metric')
    plt.ylim(y_lim[0], y_lim[1])
    plt.legend()
    plt.grid()
