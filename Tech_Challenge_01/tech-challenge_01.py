#Importanto bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 

#Exploração de dados:
pd.set_option('display.max_columns', None)

#Carregue a base de dados e explore suas características;
read_csv = pd.read_csv('/Users/izabela.oliveira/Downloads/insurance.csv')
print(read_csv.head()) #imprime as 5 primeiras linhas

#print(read_csv.describe()) # imprime estatísticas descritivas

#print(read_csv.info()) # imprime informações sobre o dataframe

#Tratar valores nulos
#print(read_csv.isnull().sum()) # imprime a soma de valores nulos por coluna


#Tratar valores duplicados
#print(read_csv.duplicated().sum()) # imprime a soma de valores duplicados

#Analise estatísticas descritivas e visualize distribuições relevantes.
# Dividindo o conjunto de dados em treino e teste
#X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)

# Usando LabelEncoder para transformar variáveis categóricas em numéricas 
label_encoder = LabelEncoder()
data['age'] = label_encoder.fit_transform(data['age'])


# Inicializar o scaler usando apenas o conjunto de treino
scaler = MinMaxScaler()
paramenters = read_csv.columns
print (scaler.fit(paramenters))
print (scaler.data_max_)
print (scaler.data_min_)
print (scaler.transform(paramenters))

# Aplicar o Z-score nas features de treino
#X_train_scaled = scaler.transform(X_train)

#  Aplicar  o  Z-score  nas  features  de  teste  usando  as  estatísticas  do conjunto de treino
#X_test_scaled = scaler.transform(X_test)