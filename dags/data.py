from sklearn.model_selection import train_test_split
import pandas as pd
from typing import List
from config import config


def prepare_data(csv_path: str) -> List[str]:
    """Чтение загруженного датасета и разделение на train и test выборки."""
    wine = pd.read_csv(csv_path, sep=',')
    wine_train, wine_test = train_test_split(wine,
                                             test_size=config["data"]["test_size"],
                                             random_state=config["random_state"])
    wine_train.to_csv('/app/data/wine_train.csv', sep=',', index=False)
    wine_test.to_csv('/app/data/wine_test.csv', sep=',', index=False)



