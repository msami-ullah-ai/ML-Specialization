import pandas as pd
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('default.csv')
x_train = data[['age','income','credit_score']].values
y_train = data['target']

logistic_model = LogisticRegression()
logistic_model.fit(x_train, y_train)

y_predict = logistic_model.predict(x_train)
print("Prediction on training set:", y_predict)