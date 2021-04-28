# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 10:01:29 2021

@author: Diego Higuera
"""

#%% Importacion de funciones
import numpy as np
import pandas as pd
import scipy.special as sc
import matplotlib.pyplot as plt

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

def incertidumbre (q_ordenados,q_nse_ordenados,lower,upper,filas,columnas):
    q_min = np.zeros(columnas)
    q_max = np.zeros(columnas)
    for i in range(columnas):
        dist_acu = np.zeros(filas)
        acu = 0
        dist = sum(q_nse_ordenados[:,i])
        q_par = []
        if dist == 0 : continue
        for t in range (filas):
            acu = acu + q_nse_ordenados[t,i]
            dist_acu[t] = acu/dist
            
            if dist_acu[t] > 0.05 and dist_acu[t] < 0.95:
                q_par.append(q_ordenados[t,i])
        
        if q_par == [] : continue
        q_min[i] = min(q_par)
        q_max[i] = max(q_par)        
        
    return q_min, q_max

    
#%% Entrada de datos

file_name = 'Datos_entrada_prueba1.csv'                 # Archivo con datos observados
entrada_d = pd.read_csv(file_name,header=None)
entrada = entrada_d.to_numpy()

t = entrada[:,0]                                # Tiempo en días 
Q = 3456.0                                       # Caudal en m3/d
r = 38.5                                        # Radio en m

#%% Entrada de parámetros

file_name_2 = 'Datos aleatorios.csv'
entrada_p = pd.read_csv(file_name_2,header=None)
parametros = entrada_p.to_numpy()

T = parametros[:,0]
S = parametros[:,1]

#%% Variables adicionales

(x_1,y_1) = parametros.shape
(x_2,y_2) = entrada.shape
metodos = 2
v_nse = np.zeros((x_1,metodos))
serie_cp = np.zeros(x_2)
serie_th = np.zeros(x_2)
nse_max_cp = -1000
nse_max_th = -1000
opt_cp = np.array([0,0])
opt_th = np.array([0,0])
q_nse = np.zeros((1,x_2))

#%% Calibración por monte carlo 

for e in range(x_1):    

    for i in range(x_2):
        serie_cp[i] = cooper(Q,T[e],t[i],r,S[e])
        serie_th[i] = theis(u(r,S[e],T[e],t[i]),Q,T[e])
    
    nse_cp = nse(entrada[:,1],serie_cp)
    nse_th = nse(entrada[:,1],serie_th)
    
    v_nse[e,0] = nse_cp
    v_nse[e,1] = nse_th
    
    if nse_cp > nse_max_cp:
        nse_max_cp = nse_cp
        opt_cp = [T[e],S[e]]
        s_opt_cp = serie_cp
        
    if nse_th > nse_max_th:
        nse_max_th = nse_th
        opt_th = [T[e],S[e]]
        s_opt_th = serie_th
        
    if nse_cp > 0.40:
        q_nse = np.insert(q_nse,q_nse.shape[0],serie_cp,0)

#%% Imprimir mejor corrida vs observados

plt.figure(figsize=[8,4.5],dpi=400)
plt.plot(t,s_opt_cp,'r-',label='Calculado',linewidth=0.5)
plt.plot(t,entrada[:,1],label='Observado',linewidth=0.5)
plt.title('Calculado vs observado')
plt.xlabel('Tiempo (días)')
plt.ylabel('s (m)')
plt.legend()
plt.grid(b=True,linewidth=0.5)  

#%% Sensibilidad de parámetros por dotty plots

plt.figure(figsize=[8,4.5],dpi=400)
plt.scatter(T,v_nse[:,0],0.5)
plt.title('Sensibilidad')
plt.xlabel('T (m2/d)')
plt.ylabel('nse (max)')

plt.figure(figsize=[8,4.5],dpi=400)
plt.scatter(S,v_nse[:,0],0.5)
plt.title('Sensibilidad')
plt.xlabel('S')
plt.ylabel('nse (max)')

#%% Análisis de incertidumbre (GLUE)

log_nse = v_nse[:,0] > 0.40
sum_log = int(sum(log_nse))
nse_fin = v_nse[log_nse]

q_nse = np.delete(q_nse,0,0)
for i in range(sum_log):
    q_nse = np.insert(q_nse,q_nse.shape[0],q_nse[i,:]*nse_fin[i,0],0)

q_nse_2 = q_nse[:sum_log,:]
q_nse_2 = np.insert(q_nse_2,q_nse_2.shape[1],nse_fin[:,0],1)
q_nse_2 = pd.DataFrame(q_nse_2)  
q_nse_2.sort_values(x_2,ascending=False,inplace=True)
q_nse_2 = q_nse_2.to_numpy()
q_nse_1 = q_nse[sum_log:,:]
q_nse_1 = np.insert(q_nse_1,q_nse_1.shape[1],nse_fin[:,0],1)
q_nse_1 = pd.DataFrame(q_nse_1)  
q_nse_1.sort_values(x_2,ascending=False,inplace=True)
q_nse_1 = q_nse_1.to_numpy()

q_min, q_max = incertidumbre(q_nse_2,q_nse_1,0.05,0.95,sum_log,x_2)        
        
plt.figure(figsize=[8,4.5],dpi=400)
plt.plot(t,q_min,'r-',label='Lower',linewidth=0.5)
plt.plot(t,q_max,'b-',label='Higher',linewidth=0.5)
plt.plot(t,q_nse_2[0,:x_2],'g-',label='Best',linewidth=0.5)
plt.plot(t,entrada[:,1],'ro',label='Data',markersize=0.7)
plt.fill_between(t,q_min,q_max,color=(0.92,0.97,0.98))
plt.title('Incertidumbre')
plt.xlabel('Tiempo (d)')
plt.ylabel('s (m)')
plt.legend()
plt.grid(b=True,linewidth=0.5)    

#%% Mapas de Calor

long = 200
T_m = np.linspace(250,275,long)
S_m = np.linspace(1e-7,1e-6,long)
grid = np.zeros((long,long))

for a in range(long):
    for b in range(long):
        serie_m = np.zeros(x_2)
        for y in range(x_2):
            serie_m[y] = cooper(Q,T[a],t[y],r,S[b])
        grid[b,a] = nse(entrada[:,1],serie_m)
        
plt.figure(dpi=400)    
plt.contourf(T_m,S_m,grid,10,vmin=-1)
plt.colorbar()
plt.xlabel('T (m2/d)')
plt.ylabel('S (?)')
plt.title('Sensibilidad de parámetros')