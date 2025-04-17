#Ferramenta simples e eficiente para mineração e análise de dados
#Exemplo: treinar modelo de regressão linear
from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3

model = LinearRegression().fit(X, y)
print(model.coef_)