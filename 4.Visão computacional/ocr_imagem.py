import cv2
import matplotlib.pyplot as plt
import pytesseract


#carregar imagem
imagem = cv2.imread('/Users/izabela.oliveira/Documents/GitHub/Pos-tech/Visão computacional/pensador.png')

if imagem is None:
    print("Erro ao carregar a imagem.")
    exit()

#exibir imagem original
#cv2.imshow('Imagem Original', imagem)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#Aplicando o Tesseract OCR
#texto = pytesseract.image_to_string(imagem, lang='por')
#print("Texto extraído da imagem:\n", texto)

#Converter para grayscale
#imagem_gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

#exibir imagem grayscale
#cv2.imshow('Imagem grayscale', imagem_gray)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#Redimensionar imagem
#imagem_redimensionada = cv2.resize(imagem, (800, 600))
#exibir imagem redimensionada
#cv2.imshow('Imagem Redimensionada', imagem_redimensionada)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#Salvar imagem redimensionada
#cv2.imwrite('/Users/izabela.oliveira/Documents/GitHub/Pos-tech/Visão computacional/pensador_redimensionada.png', imagem_redimensionada)

#Exibir imagem redimensionada usando matplotlib
#plt.imshow(cv2.cvtColor(imagem_redimensionada, cv2.COLOR_BGR2RGB))
#plt.axis('off')  # Remove os eixos
#plt.show()

#Imagem suavizada
#imagem_suavizada = cv2.GaussianBlur(imagem, (15, 15), 0)

# Exibir imagem suavizada
#plt.imshow(cv2.cvtColor(imagem_suavizada, cv2.COLOR_BGR2RGB))
#plt.axis('off')  # Remove os eixos
#plt.title('Imagem Suavizada')
#plt.show()

# Aplicar o Tesseract OCR na imagem suavizada
#texto_suavizado = pytesseract.image_to_string(imagem_suavizada, lang='por')
#print("Texto extraído da imagem suavizada:\n", texto_suavizado)

#Detecção de bordas
imagem_bordas = cv2.Canny(imagem, 100, 200)

# Exibir imagem com bordas
plt.imshow(imagem_bordas, cmap='gray')
plt.axis('off')  # Remove os eixos
plt.title('Imagem com Bordas')
plt.show()

# Aplicar o Tesseract OCR na imagem com bordas
texto_bordas = pytesseract.image_to_string(imagem_bordas, lang='por')
print("Texto extraído da imagem com bordas:\n", texto_bordas)
# Salvar imagem com bordas
cv2.imwrite('/Users/izabela.oliveira/Documents/GitHub/Pos-tech/Visão computacional/pensador_bordas.png', imagem_bordas)
