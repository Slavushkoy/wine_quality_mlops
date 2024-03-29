from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from joblib import dump, load
from config import config
from clearml import Task
import pandas as pd


def train(train_csv: str) -> str:
    """Обучение модели логистической регрессии на тренировочной выборке и сохранение модели."""
    wine_train = pd.read_csv(train_csv, sep=',')
    X_train = wine_train.drop('quality', axis=1)
    y_train = wine_train['quality']
    model = LinearRegression()
    model.fit(X_train, y_train)
    dump(model, '/app/model')


def test(test_csv, **kwargs) -> str:
    """Тестирование модели на тестовой выборке и сохранение результатов."""
    model = load('/app/model')
    wine_test = pd.read_csv(test_csv, sep=',')
    X_test = wine_test.drop('quality', axis=1)
    y_test = wine_test['quality']

    # Предсказание на тестовом наборе
    y_pred = model.predict(X_test)

    # Оценка качества модели
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Инициализация задачи
    task = Task.init(project_name='mlops_project', task_name='liner_model')
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
