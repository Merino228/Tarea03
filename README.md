# Tarea03
Tarea03 Modelos Probabilisticos 

Para la solucion de la Tarea03 se trabajó con los archivos *.csv*  brindados. Por lo que se comenzó con la importación de las librerias.



    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits import mplot3d
    import pandas as pd
    import numpy as np

Importamos la informacion con *Pandas* en un *DataFrame*

    datos = pd.read_csv('xy.csv')
    df=pd.DataFrame(datos)
    df1=df.drop(columns=['Unnamed: 0'])

    datos1 = pd.read_csv('xyp.csv')
    df2 =pd.DataFrame(datos1)

Definimos dos vectores *X* y *Y* con los valores dados.

    y = np.arange(5,26,1)
    x = np.arange(5,16,1)


## Pregunta 1.


