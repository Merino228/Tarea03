# Tarea03
Tarea03 Modelos Probabilisticos 

Para la solucion de la Tarea03 se trabajó con los archivos *.csv*  brindados. Por lo que se comenzó con la importación de las librerias.



    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits import mplot3d
    import pandas as pd
    import numpy as np

Importamos la información con *Pandas* en un *DataFrame*

    datos = pd.read_csv('xy.csv')
    df=pd.DataFrame(datos)
    df1=df.drop(columns=['Unnamed: 0'])

    datos1 = pd.read_csv('xyp.csv')
    df2 =pd.DataFrame(datos1)

Definimos dos vectores *X* y *Y* con los valores dados.

    y = np.arange(5,26,1)
    x = np.arange(5,16,1)


## Pregunta 1.
Primero se deben calcular las funciones de densidad marginales de *x* y *y* , para ello se realiza la suma de la siguiente forma:

    # Funciones Marginales.
    fx = np.sum(df, axis=1)
    fy = np.sum(df1, axis=0)

Se observa que ambas funciones marginales tienen una distribución Gaussiana, por lo que se calculará el ajuste Gaussiano a cada una de ellas.
Se crean dos vectores ( uno para *X* y otro para *Y* ) donde se crearan los ajustes para cada variable aleatoria.

    xs = np.linspace(5,15,110000)
    ys = np.linspace(5,25,110000)

Se define la función de distribución Gaussiana:

    def gaussiana(x, mu, sigma):
        return 1/(np.sqrt(2*np.pi*sigma**2)) * np.exp(-(x - mu)**2/(2*sigma**2))

Donde dicha función se utiliza en conjunto con el comando *curve_fit* para obtener los parámetros correspondientes a dicha función.

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

Donde los parametros obtenidos son:

![alt text](https://github.com/Merino228/Tarea03/blob/master/mu1.gif)  
![alt text](https://github.com/Merino228/Tarea03/blob/master/mu2.gif)  
![alt text](https://github.com/Merino228/Tarea03/blob/master/sigma1.gif)  
![alt text](https://github.com/Merino228/Tarea03/blob/master/Sigma2.gif)  

Finalmente se obtiene el ajuste para cada función marginal con los parámetros obtenidos.

    # Ajuste de la Funcion Marginal
    ajusteX = gaussiana(xs,muX,sigmaX)
    ajusteY = gaussiana(ys,muY,sigmaY)


## Pregunta 2. 
Como ambas funciones tenian una distribución Gaussiana, se puede calcular la expresión de densidad conjunta multiplicando cada distribución gausiana, obteniendo asi la siguiente expresión:

![alt text](https://github.com/Merino228/Tarea03/blob/master/Pregunta2.gif)

## Pregunta 3. 

Para obtener los valores de Correlacion, covarianza y el coeficiente de correlacion se definieron las siguientes funciones:

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

Donde se obtiene los siguientes resultados:
* *La correlacion de X y de Y es 149.5428*, Note que existe una alta correlacion entre los valores de *X* y de *Y*

* *La covarianza de X y de Y es 0.06669*, Note que para realizar los calculos se asumio previamente que existe independencia entre *X* y *Y* por lo que el valor de la covarianza debe de ser cercano a 0.

* *El coeficiente de correlación de X y de Y es 0.003353*, Note que este coeficiente representa una dependencia lineal entre las dos variables aleatorias, cuando este coeficiente es cercano a cero se concluye que no existe una dependencia lineal entre las variables, no obstante puede existir una dependencia no lineal. 

## Pregunta 4.

Por ultimo, se grafico todos los datos obtenidos anteriormente.  
#### Datos obtenidos  
![alt text](https://github.com/Merino228/Tarea03/blob/master/Datos.png)  
#### Funcion de densidad marginal de *X*   
![alt text](https://github.com/Merino228/Tarea03/blob/master/VariableX.png)  
#### Funcion de densidad marginal de *Y*   
![alt text](https://github.com/Merino228/Tarea03/blob/master/VariableY.png)  
#### Funcion de densidad Conjunta  
![alt text](https://github.com/Merino228/Tarea03/blob/master/DensidadConjunta.png)  

