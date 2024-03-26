from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from data import get_data
from clearml import Task


def train(model, x_train, y_train) -> None:
    """Обучение модели логистической регрессии на тренировочной выборке."""
    model.fit(x_train, y_train)


def test(model, x_test, y_test) -> None:
    """Тестирование модели на тестовой выборке и сохранение результатов."""
    # Предсказание на тестовом наборе
    y_pred = model.predict(x_test)

    # Оценка качества модели
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Инициализация задачи
    task = Task.init(project_name='mlops_project', task_name='logistic_regression_model')
    task.connect({})
    logger = task.get_logger()
    logger.report_single_value(name='mse ', value=mse)
    logger.report_single_value(name='r2', value=r2)
    logger.report_single_value(name='mae', value=mae)
    task.close()

    print(f'Mean Squared Error: {mse}')
    print(f'R^2: {r2}')
    print(f'Mean Absolute Error: {mae}')


if __name__ == "__main__":
    logistic_regression_model = LinearRegression()
    data = get_data()
    train(logistic_regression_model, data["x_train"], data["y_train"])
    test(logistic_regression_model, data["x_test"], data["y_test"])



