import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
import yaml
import pickle
import os

def train_model():
    # Чтение параметров
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    model_params = params['train']
    
    # Загрузка данных
    train_data = pd.read_csv('data/processed/train.csv')
    test_data = pd.read_csv('data/processed/test.csv')
    
    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]
    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]
    
    # Настройка MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("iris_classification")
    
    with mlflow.start_run():
        # Логирование параметров
        mlflow.log_params(model_params)
        
        # Обучение модели
        model = LogisticRegression(
            max_iter=model_params['max_iter'],
            random_state=model_params['random_state']
        )
        model.fit(X_train, y_train)
        
        # Предсказания и метрики
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Логирование метрик
        mlflow.log_metric("accuracy", accuracy)
        
        # Сохранение модели
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        # Логирование артефактов
        mlflow.log_artifact('model.pkl')
        mlflow.log_artifact('params.yaml')
        
        # Логирование модели в MLflow
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Model training completed! Accuracy: {accuracy:.4f}")
        
        # Вывод отчета классификации
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    train_model()