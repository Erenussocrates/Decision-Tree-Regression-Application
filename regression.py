from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import data_processing_interface as dpi

data_df = dpi.df

RS_number = 42

X = data_df.drop(['sinif'], axis=1)
y = data_df['sinif']
#Bağımlı ve bağımsız değişkenleri oluşturuyoruz

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = RS_number)

print("Feature'larımızın train ve test kümelerinin şekilleri:\n", X_train.shape, X_test.shape)
#Output:  (1157, 6) (571, 6)

print("Feature train kümesinin data typeları:\n", X_train.dtypes)
print("Feature train kümesi head'i:\n",X_train.head())

model = DecisionTreeRegressor()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R2 skoru: ", r2_score(y_test, y_pred))

# Visualize the results
plt.figure(figsize=(10, 6))

# Scatter plot of actual test values
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual y_test')

# Line plot of predicted values
plt.plot(range(len(y_pred)), y_pred, color='red', linewidth=2, label='Predicted y_pred')

plt.title('Actual vs Predicted values')
plt.xlabel('Test Sample Index')
plt.ylabel('Sinif')
plt.legend()
plt.show()

"""
r2 skoruna bakılırsa 84%'lük bir accuracy var.
"""