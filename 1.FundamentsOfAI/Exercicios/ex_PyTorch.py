#Exemplo: treinar um modelo de rede neural simples
import torch
import torch.nn as nn
import torch.optim as optim

# Exemplo de dados de entrada
# X: matriz com 100 amostras e 3 características
# y: vetor de saída com 100 valores
X = torch.randn(100, 3)  # Gera uma matriz de 100x3 com valores aleatórios
y = torch.randn(100, 1)  # Gera um vetor de 100x1 com valores aleatórios

class SimpleNN(nn.Module):
   def __init__(self):
       super(SimpleNN, self).__init__()
       self.fc1 = nn.Linear(3, 10)
       self.fc2 = nn.Linear(10, 1)

   def forward(self, x):
       x = torch.relu(self.fc1(x))
       x = self.fc2(x)
       return x

model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
   optimizer.zero_grad()
   outputs = model(X)
   loss = criterion(outputs, y)
   loss.backward()
   optimizer.step()
   print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')