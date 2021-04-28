# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 14:43:46 2021

@author: diego
"""

#%% Importacion de funciones
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.special as sc

#%% Definicion de funciones
def u(r,S,T,t):
    return (r**2*S)/(4*T*t)

def theis (u,Q,T):
    x = Q/(4*np.pi*T)*-sc.expi(u)
    if x < 0:
        return 0
    else:
        return x

def cooper (Q,T,t,r,S):
    x = Q/(4*np.pi*T)*np.log(2.25*T*t/(r**2*S))
    if x < 0:
        return 0
    else:
        return x
    
def nse (datos_observados,datos_calculados):
    size = datos_observados.shape[0]
    den_1 = 0
    num_1 = 0
    promedio = datos_observados.mean()
    for i in range(size):
        den_1 = den_1 + (datos_observados[i]-promedio)**2
        num_1 = num_1 + (datos_observados[i]-datos_calculados[i])**2    
    return 1 - (num_1/den_1)


#%% Entrada de datos

file_name = 'Datos_entrada_prueba1.csv'                 # Archivo con datos observados
entrada_d = pd.read_csv(file_name,header=None)
entrada = entrada_d.to_numpy()

t = entrada[:,0]                                # Tiempo en días 
Q = 3456.0                                       # Caudal en m3/d
r = 38.5                                        # Radio en m

#%% Entrada de parámetros

lista = []
lista_f = []
for letra in open('in.dat'):
    letra = letra.lstrip()
    lista = letra.split()
    lista_f.append(float(lista[0]))

T = lista_f [0]
S = lista_f [1]

#%% Variables adicionales

(x_2,y_2) = entrada.shape
serie_cp = np.zeros(x_2)

#%% Calculo

for i in range(x_2):
    serie_cp[i] = cooper(Q,T,t[i],r,S)
    
v_nse = nse(entrada[:,1],serie_cp)
    
#%% Salida de datos

salida = open('out.dat','wt')
for i in range(x_2):
    salida.write(str(format(serie_cp[i],'.10f'))+'\n')
salida.close()

#%% Imprimir resultados

# =============================================================================
# plt.figure(figsize=[8,4.5],dpi=400)
# plt.plot(t,serie_cp,'r-',label='Calculado',linewidth=0.5)
# plt.plot(t,entrada[:,1],label='Observado',linewidth=0.5)
# plt.title('Calculado vs observado')
# plt.xlabel('Tiempo (días)')
# plt.ylabel('s (m)')
# plt.legend()
# plt.grid(b=True,linewidth=0.5)  
# =============================================================================
