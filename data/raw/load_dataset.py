
from sklearn.datasets import load_iris
import pandas as pd


iris = load_iris()


iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)


iris_df['target'] = iris.target
iris_df['species'] = iris_df['target'].apply(lambda x: iris.target_names[x])


iris_df.to_csv('data.csv', index=False)

print("Датасет Iris сохранен в файл 'data.csv' в текущей папке.")