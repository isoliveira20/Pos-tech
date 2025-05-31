#Importanto bibliotecas
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 

#Exploração de dados:
pd.set_option('display.max_columns', None)

#Carregue a base de dados e explore suas características;
df = pd.read_csv('/Users/izabela.oliveira/Downloads/insurance.csv')
print(df.head()) #imprime as 5 primeiras linhas

print(df.describe()) #imprime estatísticas descritivas

print(df.info()) # imprime informações sobre o dataframe

#Tratar valores nulos
print(df.isnull().sum()) # imprime a soma de valores nulos por coluna

#Verifica se existem valores nulos
print(df.isnull().values.any()) # imprime True se houver valores nulos, caso contrário False


#Tratar valores duplicados
print(df.duplicated().sum()) # imprime a soma de valores duplicados

#Visualizar a distribuição das variáveis
sns.pairplot(df)
plt.show()


# Usando LabelEncoder para transformar variáveis categóricas em numéricas 
label_encoder = LabelEncoder()
for col in ['sex', 'smoker', 'region']:
    df[col] = label_encoder.fit_transform(df[col])

print(df.head()) # Verificando a transformação

# Dividir os dados em variáveis features (X) e target (y)
x = df.drop('charges', axis=1)  # Dados de entrada - todas as colunas menos 'charges'
y = df['charges']  # A coluna 'charges' é o target, variável que queremos prever

# Dividindo o conjunto de dados em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=42)

# Dados de treino - Normalizar os dados numéricos (colocar tudo na mesma escala — geralmente de 0 a 1)
scaler = MinMaxScaler() # Inicializa o MinMaxScaler
features = df.drop('charges', axis=1).columns # Nome das colunas dos dados de entrada
X_train_scaled = scaler.fit_transform(X_train)  # aprende (fit) e aplica (transform) no treino
X_test_scaled = scaler.transform(X_test)        # aprende (fit) e aplica (transform) a mesma escala no teste
df_scaled = scaler.fit_transform(df[features]) # Cria um novo DataFrame com os dados normalizados

# Visualizando a distribuição das variáveis
plt.figure(figsize=(10, 6))
sns.histplot(df['charges'], kde=True)
plt.xlabel('Charges')
plt.title('Distribuição da variável dependente')
plt.show()
# Visualizando a distribuição das variáveis
plt.figure(figsize=(10, 6))
sns.histplot(df['age'], kde=True)
plt.xlabel('Idade')
plt.title('Distribuição da variável dependente')
plt.show()
# Visualizando a distribuição das variáveis
plt.figure(figsize=(10, 6))
sns.histplot(df['bmi'], kde=True)
plt.xlabel('IMC')
plt.title('Distribuição da variável dependente')
plt.show()
# Visualizando a distribuição das variáveis
plt.figure(figsize=(10, 6))
sns.histplot(df['children'], kde=True)
plt.xlabel('Número de filhos')
plt.title('Distribuição da variável dependente')
plt.show()
# Visualizando a distribuição das variáveis
plt.figure(figsize=(10, 6))
sns.histplot(df['smoker'], kde=True)
plt.xlabel('Fumantes')
plt.title('Distribuição da variável dependente')
plt.show()

# Visualizando a matriz de correlação
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Matriz de correlação')
plt.show()

# Inicializando o modelo de regressão linear
model = LinearRegression()

# Treinando o modelo
model.fit(X_train, y_train)

# Realizando previsões
y_pred = model.predict(X_test)

# Avaliando o modelo
mae = mean_absolute_error(y_test, y_pred) # calcula o erro absoluto médio
mse = mean_squared_error(y_test, y_pred) # calcula o erro quadrático médio
rmse = np.sqrt(mse) # calcula a raiz do erro quadrático médio
r2 = r2_score(y_test, y_pred) # calcula o coeficiente de determinação R²

# Imprimindo os resultados
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
print(f'R2: {r2}')

# Visualizando os resultados
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel('Valores reais')
plt.ylabel('Valores previstos')
plt.title('Valores reais vs Valores previstos')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.show()

# Visualizando os resíduos
plt.figure(figsize=(10, 6))
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.xlabel('Valores previstos')
plt.ylabel('Resíduos')
plt.title('Resíduos vs Valores previstos')
plt.axhline(0, color='k', lw=2)
plt.show()

# Visualizando os resíduos
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30)
plt.xlabel('Resíduos')
plt.ylabel('Frequência')
plt.title('Distribuição dos resíduos')
plt.show()

# Visualizando os resíduos
plt.figure(figsize=(10, 6))
plt.scatter(y_test, residuals)
plt.xlabel('Valores reais')
plt.ylabel('Resíduos')
plt.title('Resíduos vs Valores reais')
plt.axhline(0, color='k', lw=2)
plt.show()
