# Estimators

import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

df = pd.read_csv('iris.csv')

print(df.head())

y = df['target']
X = df.drop('target', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

feat_cols = []

for col in X.columns:
    feat_cols.append(tf.feature_column.numeric_column(col))

input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=10, num_epochs=5, shuffle=True)

# Deep Neural Network classifier

classifier = tf.estimator.DNNClassifier(hidden_units=[10, 20, 10], n_classes=3, feature_columns=feat_cols)

# Training the model

classifier.train(input_fn=input_func, steps=50)

prediction_fn = tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=len(X_test), shuffle=False)

predictions = list(classifier.predict(input_fn=prediction_fn))

final_predictions = []

for prediction in predictions:
    final_predictions.append(prediction['class_ids'][0])

print(confusion_matrix(y_test, final_predictions))

print(classification_report(y_test, final_predictions))
