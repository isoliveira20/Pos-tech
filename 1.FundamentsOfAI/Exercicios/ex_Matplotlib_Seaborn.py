#Biblioteca padrão para visualização de dados. Permite gráficos em 2D de alta qualidade. Inclui gráficos de dispersão, linhas, barras, histogramas etc.
import matplotlib.pyplot as plt
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]
plt.plot(x, y)
plt.show()

#Biblioteca de visualização baseada em Matplotlib. Permite gráficos estatísticos mais complexos e bonitos. Gráficos complexos,como mapas de calor e gráficos de distruibuição, com menos código e melhores opções de estética e estilo.
import seaborn as sns
import matplotlib.pyplot as plt
data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
sns.histplot(data)
plt.show()