from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from joblib import dump
from data import get_data
from config import config
from clearml import Task


def train(model, x_train, y_train) -> None:
    """Обучение CatBoostRegressor на тренировочной выборке."""
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
    task = Task.init(project_name='mlops_project', task_name='catboost_model')
    task.connect(config)
    logger = task.get_logger()
    logger.report_single_value(name='mse ', value=mse)
    logger.report_single_value(name='r2', value=r2)
    logger.report_single_value(name='mae', value=mae)
    dump(model, "model2.pkl", compress=True)
    task.close()

    print(f'Mean Squared Error: {mse}')
    print(f'R^2: {r2}')
    print(f'Mean Absolute Error: {mae}')


if __name__ == "__main__":
    catboost_model = CatBoostRegressor(
        max_depth=config["catboost"]["max_depth"],
        iterations=config["catboost"]["iterations"],
        l2_leaf_reg=config["catboost"]["l2_leaf_reg"],
        learning_rate=config["catboost"]["learning_rate"],
        random_state=config["random_state"]
    )
    data = get_data()
    train(catboost_model, data["x_train"], data["y_train"])
    test(catboost_model, data["x_test"], data["y_test"])
