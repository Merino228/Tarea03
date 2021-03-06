# Adrian Merino

# Tarea 03

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import scipy.interpolate as interpld
import fitter
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d

datos = pd.read_csv('xy.csv')
df=pd.DataFrame(datos)
df1=df.drop(columns=['Unnamed: 0'])

datos1 = pd.read_csv('xyp.csv')
df2 =pd.DataFrame(datos1)
print(df2)

y = np.arange(5,26,1)
x = np.arange(5,16,1)

x3d = df2['x']
y3d = df2['y']
p3d = df2['p']


# Pregunta 1. Encontrar la mejor curva de ajuste (modelo probabilístico) para las funciones de densidad marginales de X y Y.

xs = np.linspace(5,15,110000)
ys = np.linspace(5,25,110000)

# Funciones Marginales.
fx = np.sum(df, axis=1)
fy = np.sum(df1, axis=0)


# Ambas son Gaussianas.
def gaussiana(x, mu, sigma):
    return 1/(np.sqrt(2*np.pi*sigma**2)) * np.exp(-(x - mu)**2/(2*sigma**2))

# Parametros para Gaussiana
Xparam, _ = curve_fit(gaussiana,x, fx)
muX = Xparam[0]
print('mu para X ',muX)
sigmaX = Xparam[1]
print('Sigma para X ',sigmaX)

Yparam, _ = curve_fit(gaussiana,y, fy)
muY = Yparam[0]
print('mu para Y ',muY)
sigmaY = Yparam[1]
print('Sigma para Y ',sigmaY)

# Ajuste de la Funcion Marginal

ajusteX = gaussiana(xs,muX,sigmaX)
ajusteY = gaussiana(ys,muY,sigmaY)


# Pregunta 2. Asumir independencia de X y Y. Analíticamente, ¿cuál es entonces la expresión de la función de densidad conjunta que modela los datos?

# Pregunta 3. Hallar los valores de correlación, covarianza y coeficiente de correlación (Pearson) para los datos y explicar su significado.

correlacion = 0
covarianza = 0
coeficiente = 0

# Correlacion
for i in range (0,231):
    val1 = df2.iloc[i]
    mult1 = val1[0]*val1[1]*val1[2]
    correlacion = mult1 + correlacion
# Covarianza
for i in range (0,231):
    val2 = df2.iloc[i]
    mult2 = (val2[0]-muX)*(val2[1]-muY)*val2[2]
    covarianza = mult2 + covarianza
# Coeficiente de Correlacion
for i in range (0,231):
    val3 = df2.iloc[i]
    mult3 = ((val3[0]-muX)/sigmaX)*((val3[1]-muY)/sigmaY)*val3[2]
    coeficiente = mult3 + coeficiente
print('La correlacion de X y de Y es ',correlacion)
print('La covarianza de X y de Y es',covarianza)
print('El  coeficiente de correlación de X y de Y es',coeficiente)

# Pregunta 4. Graficar las funciones de densidad marginales (2D), la función de densidad conjunta (3D).

# Datos
plt.plot(x, fx)
plt.plot(y, fy)
plt.title('Datos')
plt.xlabel('Valor')
plt.ylabel('Probabilidad ')
plt.show()

# Ajuste X
plt.plot(xs,ajusteX)
plt.plot(x,fx)
plt.title('Probabilidad Marginal y ajuste para Variable X')
plt.xlabel('Valor X')
plt.ylabel('Probabilidad')
plt.show()

# Ajuste Y
plt.plot(ys, ajusteY)
plt.plot(y,fy)
plt.title('Probabilidad Marginal y ajuste para Variable Y')
plt.xlabel('Valor Y')
plt.ylabel('Probabilidad')
plt.show()

# 3D
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(x3d, y3d, p3d, c=p3d, cmap='viridis', linewidth=0.5)
ax.set_title('Probabilidad de Densidad Conjunta')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('P')
ax.view_init(30, -130)
plt.show()
