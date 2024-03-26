from sklearn.model_selection import train_test_split
import pandas as pd
from typing import List
from config import config


def load_dataset(csv_path) -> tuple[list]:
    wine = pd.read_csv(csv_path, sep=',')
    X = wine.drop('quality', axis=1)
    y = wine['quality']
    return X, y


def prepare_data(X, y) -> dict:
    """Чтение загруженного датасета и разделение на train и test выборки."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["data"]["test_size"],
        random_state=config["random_state"]
    )
    data = {
        "x_train": X_train,
        "x_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }
    return data


def get_data() -> dict:
    return prepare_data(*load_dataset('data/winequality-red.csv'))
