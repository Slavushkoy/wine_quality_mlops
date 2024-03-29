from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from joblib import dump, load
from config import config
from clearml import Task
import pandas as pd


def train(train_csv) -> None:
    """Обучение CatBoostRegressor на тренировочной выборке."""
    wine_train = pd.read_csv(train_csv, sep=',')
    x_train = wine_train.drop('quality', axis=1)
    y_train = wine_train['quality']
    model = CatBoostRegressor(
        max_depth=config["catboost"]["max_depth"],
        iterations=config["catboost"]["iterations"],
        l2_leaf_reg=config["catboost"]["l2_leaf_reg"],
        learning_rate=config["catboost"]["learning_rate"],
        random_state=config["random_state"]
    )
    model.fit(x_train, y_train)
    dump(model, '/app/model2')


def test(test_csv, **kwargs) -> None:
    # Тестирование модели на тестовой выборке и сохранение результатов.
    wine_test = pd.read_csv(test_csv, sep=',')
    x_test = wine_test.drop('quality', axis=1)
    y_test = wine_test['quality']
    model = load('/app/model2')
    y_pred = model.predict(x_test)

    # Оценка качества модели
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Инициализация задачи
    task = Task.init(project_name='mlops_project', task_name='catboost_model')
    task.connect(config)
    logger = task.get_logger()
    logger.report_single_value(name='mse', value=mse)
    logger.report_single_value(name='r2', value=r2)
    logger.report_single_value(name='mae', value=mae)
    task.close()

    test_results = {'Mean Squared Error': mse,
                    'R^2': r2,
                    'Mean Absolute Error': mae}
    kwargs['ti'].xcom_push(key='test_results', value=test_results)


def check_result(**kwargs):
    # Проверка условия
    test_results = kwargs['ti'].xcom_pull(key='test_results')
    try:
        old_result = pd.read_csv('/app/data/result.csv', sep=',')
        if old_result['Mean Absolute Error'][0] > test_results['Mean Absolute Error']:
            result = pd.DataFrame([test_results], index=['Metrics'])
            result.to_csv('/app/data/result.csv', sep=',')
            return 'run_dvc_push'
        else:
            return 'task_to_skip'
    except FileNotFoundError:
        result = pd.DataFrame([test_results], index=['Metrics'])
        result.to_csv('/app/data/result.csv', sep=',')
        return 'run_dvc_push'


