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

def coef_r (observados,simulados):
    return np.corrcoef(observados,simulados)[0,1]**2

def mse (observados,simulados):
    return 1-np.nanmean((observados - simulados)**2)
    
def deriv_aq(fi,fim1,fi1,ti,tim1,ti1):
    a = (fi-fim1)/np.log(ti/tim1)*np.log(ti1/ti)
    b = (fi1-fi)/np.log(ti1/ti)*np.log(ti/tim1)
    c = np.log(ti/tim1)+np.log(ti1/ti)
    
    return (a+b)/c
    
#%% Entrada de datos

file_name = 'Datos_entrada_3.csv'                # Archivo con datos observados
entrada_d = pd.read_csv(file_name,header=None)
entrada = entrada_d.to_numpy()

s = entrada[:,1]
t = entrada[:,0]                                 # Tiempo en días 
Q = 2500.0                                       # Caudal en m3/d
r = 100.0                                        # Radio en m

#%% Entrada de parámetros

lower_T = 200                           # Valor inferior para T
upper_T = 1100                          # Valor superior para T
lower_S = 1e-7                          # Valor inferior para S
upper_S = 1e-3                          # Valor supoerior para T
cant = 15000                            # Cantidad de muestras

T = np.random.uniform(lower_T,upper_T,cant)
S = np.random.uniform(lower_S,upper_S,cant)

#%% Calculo de derivada 

v_deriv_3 = np.zeros(entrada.shape[0])

v_deriv_3[0] = (s[1]-s[0])/(np.log(t[1]/t[0]))
v_deriv_3[-1] = (s[-1]-s[-2])/(np.log(t[-1]/t[-2]))

for der in range(1,entrada.shape[0]-1):
    t_i = t[der]
    pr = der
    fr = der
    for tu in range(entrada.shape[0]):
        try:
            pr = pr-1
            if np.abs(np.log(t_i/t[pr]))<0.3:
                continue
            else:
                break
        except:
            break
    
    for tu in range(entrada.shape[0]):
        try:
            fr = fr+1
            if np.abs(np.log(t[fr]/t_i))<0.3:
                continue
            else:
                break
        except:
            break
    
    if pr < 0:
        pr = 0
    if fr > entrada.shape[0]-1:
        fr = entrada.shape[0]-1
        
    v_deriv_3[der] = deriv_aq(s[der],s[pr],s[fr],t[der],t[pr],t[fr])
   

#%% Validez de datos

valor_cv = 0.1

for coef in range(v_deriv_3.shape[0]):
    v_analisis = v_deriv_3[coef:]
    cv = np.std(v_analisis)/np.mean(v_analisis)
    
    if cv > valor_cv:
        continue 
    else:
        break
 
fig, axs = plt.subplots(nrows=2,ncols=2,dpi=400,figsize=[12,8])
fig.suptitle('Datos observados y derivada suavizada & \n Datos crudos vs datos seleccionados')
axs[0,0].scatter(t,s,3.0,label='Observado')
axs[0,0].scatter(t,v_deriv_3,s=20,marker='+',label='Derivada')
axs[0,0].set_xscale('log')
axs[0,0].set_ylabel('Descenso (m)')
axs[0,0].set_xlabel('Datos crudos observados y derivada suavizada \n \n ')
axs[0,0].legend()    
axs[0,0].grid(b=True,linewidth=0.5)
axs[0,1].scatter(t,s,3.0,label='Observado')
axs[0,1].scatter(t,v_deriv_3,20,marker='+',label='Derivada')
axs[0,1].set_xscale('log')
axs[0,1].set_yscale('log')
axs[0,1].set_xlabel('Datos crudos observados y derivada suavizada \n  \n ')
axs[0,1].grid(b=True,linewidth=0.5)
axs[1,0].scatter(t[coef:],s[coef:],3.0,label='Observado')
axs[1,0].scatter(t[coef:],v_deriv_3[coef:],20,marker='+',label='Derivada')
axs[1,0].set_xscale('log')
axs[1,0].set_xlabel('Datos seleccionados y derivada suavizada \n \n Tiempo (días)')
axs[1,0].set_ylabel('Descenso (m)')
axs[1,0].grid(b=True,linewidth=0.5)
axs[1,1].scatter(t[coef:],s[coef:],3.0,label='Observado')
axs[1,1].scatter(t[coef:],v_deriv_3[coef:],20,marker='+',label='Derivada')
axs[1,1].set_xscale('log')
axs[1,1].set_yscale('log')
axs[1,1].set_xlabel('Datos seleccionados y derivada suavizada \n \n Tiempo (días)')
axs[1,1].grid(b=True,linewidth=0.5)

fig, axs = plt.subplots(nrows=1,ncols=2,dpi=400,figsize=[12,5])
fig.suptitle('Curvas diagnóstico con datos seleccionados')
axs[0].plot(t[coef:],s[coef:],'b-',label='Observado',linewidth=0.5)
axs[0].plot(t[coef:],v_deriv_3[coef:],'g-',label='Derivada',linewidth=0.5)
axs[0].set_xscale('log')
axs[0].set_xlabel('Tiempo (días)')
axs[0].set_ylabel('s(m)')
axs[0].legend()    
axs[0].grid(b=True,linewidth=0.5)
axs[1].plot(t[coef:],s[coef:],'b-',label='Observado',linewidth=0.5)
axs[1].plot(t[coef:],v_deriv_3[coef:],'g-',label='Derivada',linewidth=0.5)
axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[1].set_xlabel('Tiempo (días)')
axs[1].legend()    
axs[1].grid(b=True,linewidth=0.5)

#%% Variables adicionales

x_1 = T.shape[0]
#(x_2,y_2) = entrada.shape
x_2 = s[coef:].shape[0]
fun_obj = 3    #Definir número de funciones objetivo
metodos = 2    #Definir número de métodos de calculo (theis y cooper)
v_nse = np.zeros((x_1,metodos))
serie_cp = np.zeros(x_2)
serie_th = np.zeros(x_2)
s_opt_cp = np.zeros((x_2,fun_obj))
s_opt_th = np.zeros((x_2,fun_obj))
nse_max_cp = -1000
nse_max_th = -1000
r_max_cp = -1000
r_max_th = -1000
mse_max_cp = 1000
mse_max_th = 1000
opt_cp = np.zeros((fun_obj,metodos))
opt_th = np.zeros((fun_obj,metodos))
res_cp = np.zeros((x_1,fun_obj))
res_th = np.zeros((x_1,fun_obj))
q_nse = np.zeros((1,x_2))

#%% Calibración por monte carlo 

v_entrada_s = s[coef:]

for e in range(x_1):    

    for i in range(x_2):
        serie_cp[i] = cooper(Q,T[e],t[i+coef],r,S[e])
        serie_th[i] = theis(u(r,S[e],T[e],t[i+coef]),Q,T[e])
    
    nse_cp = nse(v_entrada_s,serie_cp)
    nse_th = nse(v_entrada_s,serie_th)
    
    r_cp = coef_r(v_entrada_s,serie_cp)
    r_th = coef_r(v_entrada_s,serie_th)
    
    mse_cp = mse(v_entrada_s,serie_cp)
    mse_th = mse(v_entrada_s,serie_th)
    
    res_cp[e,:] = [nse_cp,r_cp,mse_cp]
    res_th[e,:] = [nse_th,r_th,mse_th]
    
    if nse_cp > nse_max_cp:
        nse_max_cp = nse_cp
        opt_cp[0,:] = [T[e],S[e]]
        s_opt_cp[:,0] = serie_cp[:]        
        
    if nse_th > nse_max_th:
        nse_max_th = nse_th
        opt_th[0,:] = [T[e],S[e]]
        s_opt_th[:,0] = serie_th[:]
        
    if r_cp > r_max_cp:
        r_max_cp = r_cp
        opt_cp[1,:] = [T[e],S[e]]
        s_opt_cp[:,1] = serie_cp[:]        
        
    if r_th > r_max_th:
        r_max_th = r_th
        opt_th[1,:] = [T[e],S[e]]
        s_opt_th[:,1] = serie_th[:]
        
    if mse_cp < mse_max_cp:
        mse_max_cp = mse_cp
        opt_cp[2,:] = [T[e],S[e]]
        s_opt_cp[:,2] = serie_cp[:]        
        
    if mse_th < mse_max_th:
        mse_max_th = mse_th
        opt_th[2,:] = [T[e],S[e]]
        s_opt_th[:,2] = serie_th[:]        
        
    if nse_cp > 0.70:
        q_nse = np.insert(q_nse,q_nse.shape[0],serie_cp,0)
        
fun_obj_f = np.array([[nse_max_cp,r_max_cp,mse_max_cp],[nse_max_th,r_max_th,mse_max_th]])

#%% Imprimir mejor corrida vs observado

def plotear(time,obs,opt_sim,res,T_pl,S_pl,top_fun,tx_o):  
    
    if res == 0:
        texto = 'Ajuste: Cooper Jacob'
    elif res == 1:
        texto = 'Ajuste: Theis'   
    
    texto = str(texto + '\nFunción objetivo: '+tx_o+' con valor de: ' + str("{:.3f}".format(top_fun)))
        
    texto_2 = str('Transmisividad: '+"{:.3f}".format(T_pl)+' m2/día'
                  +' || Coeficiente de almacenamiento: '+"{:.3e}".format(S_pl))
    
    fig, axs = plt.subplots(3,dpi=400,figsize=[8,10])
    fig.suptitle('Resultados Modelación'+ '\n \n' + str(texto)+'\n'+str(texto_2))
    axs[1].plot(time,opt_sim,'r-',label='Calculado',linewidth=0.8)
    axs[1].scatter(time,obs,4.0,label='Observado')
    axs[1].set_xscale('log')    
    axs[1].set_ylabel('Descenso (m)')    
    axs[1].grid(b=True,linewidth=0.5)
    axs[0].plot(time,opt_sim,'r-',label='Calculado',linewidth=0.8)
    axs[0].scatter(time,obs,4.0,label='Observado')
    axs[0].set_ylabel('Descenso (m)')  
    axs[0].grid(b=True,linewidth=0.5)
    axs[0].legend()
    axs[2].plot(time,opt_sim,'r-',label='Calculado',linewidth=0.8)
    axs[2].scatter(time,obs,4.0,label='Observado')
    axs[2].set_xscale('log')
    axs[2].set_yscale('log')
    axs[2].set_xlabel('Tiempo (Días)')
    axs[2].set_ylabel('Descenso (m)')
    axs[2].grid(b=True,linewidth=0.5)

"""
Para plotear:
    
    Cambie la variable n_fun_obj de la siguiente forma:
        NSE: 0
        R^2: 1
        MSE: 2

"""
n_fun_obj = 0

if n_fun_obj == 0:
    texto_obj = 'NSE'
elif n_fun_obj == 1:
    texto_obj = 'R^2'
elif n_fun_obj == 2:
    texto_obj = 'MSE'

a_p = np.where(fun_obj_f[:,n_fun_obj] == np.amax(fun_obj_f[:,n_fun_obj]))[0][0]
max_fun = np.max(fun_obj_f[:,n_fun_obj])

if a_p == 0:    
    T_pl = opt_cp[n_fun_obj,0]
    S_pl = opt_cp[n_fun_obj,1]
    plotear(t[coef:],s[coef:],s_opt_cp[:,n_fun_obj],a_p,T_pl,S_pl,max_fun,texto_obj)
elif a_p == 1:    
    T_pl = opt_th[n_fun_obj,0]
    S_pl = opt_th[n_fun_obj,1]
    plotear(t[coef:],s[coef:],s_opt_th[:,n_fun_obj],a_p,T_pl,S_pl,max_fun,texto_obj)
    


#%% Sensibilidad de parámetros por dotty plots

def plot_dotty(T,S,fun):
    fig, axs = plt.subplots(nrows=1,ncols=2,sharey=True,dpi=400,figsize=[8,4.5])
    fig.suptitle('Sensibilidad de parámetros')
    axs[0].scatter(T,fun,0.5)
    axs[0].set_xlabel('Transmisividad (m2/d)')
    axs[0].set_ylabel('Función objetivo')
    axs[1].scatter(S,fun,0.5)
    axs[1].set_xlabel('Coeficiente de almacenamiento')    

n_fun = 0

plot_dotty(T,S,res_cp[:,n_fun])

#%% Análisis de incertidumbre (GLUE)

log_nse = res_cp[:,0] > 0.70
sum_log = int(sum(log_nse))
nse_fin = res_cp[:,0][log_nse]

q_nse = np.delete(q_nse,0,0)
for i in range(sum_log):
    q_nse = np.insert(q_nse,q_nse.shape[0],q_nse[i,:]*nse_fin[i],0)

q_nse_2 = q_nse[:sum_log,:]
q_nse_2 = np.insert(q_nse_2,q_nse_2.shape[1],nse_fin[:],1)
q_nse_2 = pd.DataFrame(q_nse_2)  
q_nse_2.sort_values(x_2,ascending=False,inplace=True)
q_nse_2 = q_nse_2.to_numpy()
q_nse_1 = q_nse[sum_log:,:]
q_nse_1 = np.insert(q_nse_1,q_nse_1.shape[1],nse_fin[:],1)
q_nse_1 = pd.DataFrame(q_nse_1)  
q_nse_1.sort_values(x_2,ascending=False,inplace=True)
q_nse_1 = q_nse_1.to_numpy()

q_min, q_max = incertidumbre(q_nse_2,q_nse_1,0.05,0.95,sum_log,x_2)

tiempo_plt = t[coef:]        
        
plt.figure(figsize=[8,4.5],dpi=400)
plt.plot(tiempo_plt,q_min,'r-',label='Lower',linewidth=0.5)
plt.plot(tiempo_plt,q_max,'b-',label='Higher',linewidth=0.5)
plt.plot(tiempo_plt,q_nse_2[0,:x_2],'g-',label='Best',linewidth=0.5)
plt.plot(tiempo_plt,s[coef:],'ro',label='Data',markersize=0.7)
plt.fill_between(tiempo_plt,q_min,q_max,color=(0.92,0.97,0.98))
plt.title('Incertidumbre')
plt.xlabel('Tiempo (d)')
plt.ylabel('Descenso (m)')
plt.legend()
plt.grid(b=True,linewidth=0.5)    

#%% Mapas de Calor

long = 200
T_m = np.linspace(lower_T,upper_T,long)
S_m = np.linspace(lower_S,2e-5,long)
grid = np.zeros((long,long))

for a in range(long):
    for b in range(long):
        serie_m = np.zeros(x_2)
        for y in range(x_2):
            serie_m[y] = cooper(Q,T_m[a],t[y],r,S_m[b])
        grid[b,a] = nse(s[coef:],serie_m)

rango = np.linspace(-1,1,21)

plt.figure(dpi=600,figsize=[15,4.5])    
plt.contourf(T_m,S_m,grid,rango)
plt.colorbar()
plt.xlabel('Transmisividad (T) (m2/d)')
plt.ylabel('Coeficiente de almacenamiento (S)')
plt.title('Sensibilidad de parámetros')


#%% Análisis de efectos 

dif_efecto = np.mean(s_opt_th[:,0] - s[coef:])

texto_efecto = 'no hay efecto piel o de almacenamiento'
texto_grph = 'Efecto observado: Ninguno'

if dif_efecto > 0.2:
    texto_efecto = 'se presenta efecto de almacenamiento'
    texto_grph = 'Efecto observado: Almacenamiento'
elif dif_efecto < -0.2:
    texto_efecto = 'se presenta efecto tipo piel'
    texto_grph = 'Efecto observado: Piel'
    
plt.figure(dpi=600,figsize=[9,4.5])
plt.scatter(t[coef:],s[coef:],s=3,c='r',label='Observado')
plt.plot(t[coef:],s_opt_th[:,0],label='Calculado Theis',linewidth=0.5)
plt.xlabel('Tiempo (Días)')
plt.ylabel('Descenso (m)')
plt.text(t[-4],s[coef:][2],texto_grph,ha='left',va='top')
plt.legend()
plt.grid(b=True,linewidth=0.5)    

#%% Ensayos de recuperación

""" A continuación se trabaja con otro grupo de datos asociados a 
ensayos de recuperación """

 
file_name = 'Datos_entrada_r1.csv'                # Archivo con datos observados
entrada_dr = pd.read_csv(file_name,header=None)
entrada_r = entrada_dr.to_numpy()

s_r = entrada_r[:,1]
t_r = entrada_r[:,0]                                 # Tiempo en días 
Q_r = 2500.0                                       # Caudal en m3/d
r_r = 100.0                                        # Radio en m

### Determinación de ts 

t_s = 0

for recu in range(s_r.shape[0]-1):
    if s_r[recu+1] - s_r[recu] > 0:
        continue
    else:
        t_s = t_r[recu]
        p_ts = recu
        break
        
t_rs = t_r[recu+1:]/(t_r[recu+1:]-t_s)

c_recu = np.polyfit(np.log10(t_rs),s_r[recu+1:],1)
m_recu = c_recu[0]
T_recu = 2.303*Q/(4*np.pi*m_recu)
texto_r = str('Ensayo de recuperación \n Transmisividad: '+str("{:.3f}".format(T_recu)))

s_estimada = np.polyval(c_recu,np.log10(t_rs))

fig, axs = plt.subplots(1,dpi=400,figsize=[9,4.5])
fig.suptitle(texto_r)
axs.scatter(t_rs,s_r[recu+1:],1.0,label='Observado')
axs.plot(t_rs,s_estimada,'r-',label='Aproximación',linewidth=0.8)
axs.set_xscale('log')    
axs.set_ylabel('Descenso (m)')
axs.set_xlabel('t/(t-ts)')         
axs.grid(b=True,linewidth=0.5)
axs.legend()

#%% Resumen corrida 

print('\n---- Resultados: Datos seleccionados para la calibración ----')
print('Cv inferior a ',str(valor_cv),' a partir del dato',coef+1)
print('Ver figura 1 para verificar selección de datos')

print('\n---- Resultados: Calibración del modelo ----')
 
if a_p == 0:
    res_cal = 'Mejor valor de ajuste con Cooper Jacbob'
if a_p == 1:
    res_cal = 'Mejor valor de ajuste con Theis'
    
print(res_cal, 'utilizando la función objetivo: ',texto_obj)
print('Ver figura 2 para resultados de calibración')

print('\n---- Resultados: Sensibilidad de parámetros----')
print('Ver figura 3 y 5 para identificar sensibilidad de parámetros')

print('\n----Resultados: Análisis de incertidumbre----')
print('Ver figura 4 para identificar la incertidumbre del modelo')

print('\n---- Resultados: Efectos asociados ----')
print("""Comparando los resultados de la modelación por theis con 
los datos observados se puede concluir que""",texto_efecto)
print('Ver figura 6')

print('\n---- Resultados: Ensayos de recuperación----')
print('Ver figura 7 para mayor información')

        
