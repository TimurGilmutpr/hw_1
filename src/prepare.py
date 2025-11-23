import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import os

def prepare_data():
    # Чтение параметров
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    test_size = params['prepare']['test_size']
    random_state = params['prepare']['random_state']
    
    # Загрузка данных
    data = pd.read_csv('data/raw/data.csv')
    
    # Предполагаем, что последний столбец - целевая переменная
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Сохранение
    os.makedirs('data/processed', exist_ok=True)
    pd.concat([X_train, y_train], axis=1).to_csv('data/processed/train.csv', index=False)
    pd.concat([X_test, y_test], axis=1).to_csv('data/processed/test.csv', index=False)
    
    print("Data preparation completed!")

if __name__ == "__main__":
    prepare_data()