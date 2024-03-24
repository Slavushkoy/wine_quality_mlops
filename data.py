from sklearn.model_selection import train_test_split
import pandas as pd
from typing import List


def prepare_data(csv_path: str) -> List[str]:
    """Чтение загруженного датасета и разделение на train и test выборки."""
    wine = pd.read_csv(csv_path, sep=',')
    wine_train, wine_test = train_test_split(wine, test_size=0.2, random_state=1)
    wine_train.to_csv('data/wine_train.csv', sep=',', index=False)
    wine_test.to_csv('data/wine_test.csv', sep=',', index=False)


# prepare_data("data/winequality-red.csv")