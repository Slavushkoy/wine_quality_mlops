from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from joblib import dump, load
import pandas as pd


def train(train_csv: str) -> str:
    """Обучение модели логистической регрессии на тренировочной выборке и сохранение модели."""
    wine_train = pd.read_csv(train_csv, sep=',')
    X_train = wine_train.drop('quality', axis=1)
    y_train = wine_train['quality']
    model = CatBoostRegressor()
    model.fit(X_train, y_train)
    dump(model, 'model2.pkl')


def test(model_path: str, test_csv: str) -> str:
    """Тестирование модели на тестовой выборке и сохранение результатов."""
    model = load(model_path)
    wine_test = pd.read_csv(test_csv, sep=',')
    X_test = wine_test.drop('quality', axis=1)
    y_test = wine_test['quality']

    # Предсказание на тестовом наборе
    y_pred = model.predict(X_test)

    # Оценка качества модели
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'R^2: {r2}')
    print(f'Mean Absolute Error: {mae}')


train("data/wine_train.csv")
test(model_path="model2.pkl", test_csv="data/wine_test.csv")
