import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data = np.random.randint(0, 100, (10, 2))
scaler_model = MinMaxScaler()
scaler_model.fit_transform(data)

my_data = np.random.randint(0, 101, (50, 4))
df = pd.DataFrame(data=my_data, columns=['f1', 'f2', 'f3', 'label'])

print(df)

X = df[['f1', 'f2', 'f3']]
y = df['label']

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(X_train.shape())
print(X_test.shape())
