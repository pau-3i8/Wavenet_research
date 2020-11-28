from itertools import product, combinations, groupby
from activation_funcs import phi, psi
from tqdm.auto import trange
import dask.dataframe as dd
import dask.array as da
from numba import njit
from tqdm import tqdm
import pandas as pd
import numpy as np
import psutil
import h5py

##################### Compilació numba de la funcio lstsq (només cal float64) #####################
from numpy.linalg import lstsq, eigh

#no serveix de massa.
@njit(nogil = True, fastmath = True, cache = True)
def numba_lstsq(A, b): #accelero la resolucio de min² amb numba.
    return lstsq(A, b.astype(A.dtype))[0] #vector columna
"""
print('Compilant numba per accelerar min²')
b = np.ones(2)
A = np.eye(2, dtype = np.float64)
F = numba_lstsq(A, b)
"""

########################### MATRIZ FUNCIONES DE ACTIVACIÓN 3 ENTRADAS #############################

def matriu_Fx_2D(param, input_1, input_2, Iapp):
    sf_name = param['fscale']
    n_sf = param['n_sf']
    N = len(input_1)

    ## Memòria que es farà servir:
    if param['bool_lineal']: l = 2
    else: l = 0
    if param['bool_scale']: neurones = 1
    else: neurones = 0
    for m in range(param['resolution']+1):
        neurones += (2**(3*m)+3*2**(2*m)+3*2**m)

    matriu = np.zeros((N, l+(n_sf**3)*neurones)).T
    
    ## Creas las columnas de la parte lineal
    i = 0
    if param['bool_lineal']:
        i = 2
        matriu[0] = np.ones((1, N))*input_1
        matriu[1] = np.ones((1, N))*input_2
    
    ## Creas las columnas de funciones de escala
    if param['bool_scale']:
        n = [n for n in range(n_sf)]
        for ns in list(product(n,n,n)):
            n1, n2, n3 = ns
            matriu[i] = phi(sf_name, input_1, n1)* phi(sf_name, input_2, n2)* phi(sf_name, Iapp, n3)
            i+=1

    ## Creas las columnas de wavelets
    for m in trange((param['resolution']+1), desc='Computing wavelets', unit='level', leave=False):
        n = [n for n in range(n_sf)]
        c = [c for c in range(2**m)]
        v = [(input_1, input_2, Iapp), (Iapp, input_1, input_2), (input_2, Iapp, input_1)]

        #Les K's corresponen als factors de l'equacio 6.11 del TFG amb els valors de la taula 6.2
        for elem in list(product(n,n,n)):
            n1, n2, n3 = elem
            for var in v:
                for c1 in c: #K1
                    matriu[i] = phi(sf_name, var[0], n1)* phi(sf_name, var[1], n2)* psi(sf_name, (2**m)* var[2] - c1, n3)
                    i+=1
                for ci in list(product(c,c)): #K2
                    c1, c2 = ci
                    matriu[i] = phi(sf_name, var[0], n1)* psi(sf_name, (2**m)* var[1] - c1, n2)* psi(sf_name, (2**m)* var[2] - c2, n3)
                    i+=1
            for ci in list(product(c,c,c)): #K3
                c1, c2, c3 = ci
                matriu[i] = psi(sf_name, (2**m)* input_1 - c1, n1)* psi(sf_name, (2**m)* input_2 - c2, n2)* psi(sf_name, (2**m)* Iapp - c3, n3)
                i+=1
                
    return matriu.T

########################### MATRIZ FUNCIONES DE ACTIVACIÓN 4 ENTRADAS ###########################

def matriu_Fx_3D(param, input_1, input_2, input_3, Iapp):
    sf_name = param['fscale']
    n_sf = param['n_sf']
    N = len(input_1)

    ## Memòria que es farà servir:
    if param['bool_lineal']: l = 3
    else: l = 0
    if param['bool_scale']: neurones = 1
    else: neurones = 0
    for m in range(param['resolution']+1):
        neurones += (2**(4*m)+4*2**(3*m)+6*2**(2*m)+4*2**m)

    matriu = np.zeros((N, l+(n_sf**4)*neurones)).T
        
    ## Creas las columnas de la parte lineal
    i = 0
    if param['bool_lineal']:
        i = 3
        matriu[0] = np.ones((1, N))*input_1
        matriu[1] = np.ones((1, N))*input_2
        matriu[2] = np.ones((1, N))*input_3
        
    ## Creas las columnas de funciones de escala
    if param['bool_scale']:
        n = [n for n in range(n_sf)]
        for ns in list(product(n,n,n,n)):
            n1, n2, n3, n4 = ns
            matriu[i] = phi(sf_name, input_1, n1)* phi(sf_name, input_2, n2)* phi(sf_name, input_3, n3)* phi(sf_name, Iapp, n4)
            i+=1

    ## Creas las columnas de wavelets
    for m in trange((param['resolution']+1), desc='Computing wavelets', unit='level', leave=False):
        n = [n for n in range(n_sf)]
        c = [c for c in range(2**m)]
        v1 = [(input_2, input_1, input_3, Iapp), (Iapp, input_2, input_1, input_3), (input_3, Iapp, input_2, input_1), (input_1, input_3, Iapp, input_2)]
        v2 = [(input_2, input_1, input_3, Iapp), (Iapp, input_2, input_1, input_3), (input_3, Iapp, input_2, input_1), (input_1, input_3, Iapp, input_2), (input_3, input_2, input_1, Iapp), (input_1, Iapp, input_3, input_2)]

        #Les K's corresponen als factors de l'equacio 6.11 del TFG amb els valors de la taula 6.2
        for ns in list(product(n,n,n,n)):
            n1, n2, n3, n4 = ns
            for var in v1: #K1
                for c1 in c:
                    matriu[i] = phi(sf_name, var[0], n1)* phi(sf_name, var[1], n2)* phi(sf_name, var[2], n3)* psi(sf_name, (2**m)* var[3] - c1, n4)
                    i+=1
            for var in v2: #K2
                for ci in list(product(c,c)):                
                    c1, c2 = ci
                    matriu[i] = phi(sf_name, var[0], n1)* phi(sf_name, var[1], n2)* psi(sf_name, (2**m)* var[2] - c1, n3)* psi(sf_name, (2**m)* var[3] - c2, n4)
                    i+=1
            for var in v1: #K3
                for ci in list(product(c,c,c)):
                    c1, c2, c3 = ci
                    matriu[i] = psi(sf_name, (2**m)* var[0] - c1, n1)* psi(sf_name, (2**m)* var[1] - c2, n2)* psi(sf_name, (2**m)* var[2] - c3, n3)* phi(sf_name, var[3], n4)
                    i+=1
            for ci in list(product(c,c,c,c)): #K4
                c1, c2, c3, c4 = ci
                matriu[i] = psi(sf_name, (2**m)* input_2 - c1, n1)* psi(sf_name, (2**m)* input_1 - c2, n2)* psi(sf_name, (2**m)* input_3 - c3, n3)* psi(sf_name, (2**m)* Iapp - c4, n4)
                i+=1
    
    return matriu.T

################################## FUNCIONES PARA GENERAR DATOS ##################################

### CREATE IAPPS
def generate_data(param, n_Iapps):
    Iapp = np.random.uniform(param['I_min'], param['I_max'], n_Iapps)
    Iapp = [key for key, group in groupby(Iapp) for n_times in range(param['points'])]
    return np.array(Iapp)

### IMPORT IAPPS FROM A FILE
def import_data(param):
    Iapp = read_data('inputs.txt')
    Iapp = [key for key, group in groupby(Iapp) for n_times in range(param['points'])]
    return np.array(Iapp)

## Serveix per guardar Iapps i outputs
def save_data(arxiu, vector):
    f = open(arxiu, 'w')
    [f.writelines([str(vector[i])+'\n']) for i in range(len(vector))]
    f.close()
    
## Serveix per llegir Iapps i outputs
def read_data(arxiu):
    f = open(arxiu, 'r')
    data = [float(line.split()[0]) for line in f]
    f.close()
    return data

##################################### FUNCIONES DEL ALGORITMO #####################################

### LOSS
# Ja no es farà servir hyperlearn. Necessita Fx i ara només es pot treballar amb FTF
def training(m, FX, target, mida_chunk, var):
    FTF = np.array(pd.read_parquet('FTF.parquet'))
    vaps, _ = eigh(FTF)
    mu = 1e-30
    A = FTF + np.identity(FTF.shape[1])*mu*vaps[-1].real
    
    Y = da.from_array(target, chunks = (mida_chunk, 1))
    dd.from_dask_array(da.dot(FX.T, Y), columns = ['0']).to_parquet('FTY.parquet')
    b = np.array(pd.read_parquet('FTY.parquet'))

    weights = lstsq(A, b, rcond=None)[0] #vector columna
    #weights = numba_lstsq(A, b) 
    weights = da.from_array(weights, chunks = (mida_chunk, 1))
    df = dd.from_dask_array(weights, columns = ['0'])
    df.to_parquet('weights_'+var+'.parquet')
    
    print('-> '+ var + ' MSE at level =', m+2, 'is:', MSE(FX, target, weights))

def MSE(FX, Y, weights):
    FW = dd.from_dask_array(da.dot(FX, weights), columns = ['0']).to_parquet('temp.parquet')    
    FW = np.array(pd.read_parquet('temp.parquet'))
    return np.sum((Y - FW)**2)/len(Y)

################################### Auto WN's parametrization ###################################

def descomp_factorial(num):
    factor = 2
    l = []
    while factor*factor <= num:
        if num % factor:
            factor += 1
        else:
            num = num//factor
            l.append(factor)
    if num > 1:
        l.append(num)
    return l

def producte(n):
    p = n[0]
    for elem in n[1:]:
        p *= elem
    return p

def divisors(l):
    n_factors = np.arange(1, len(l)) #combinacios de 1 a len(l)-1 factors
    ll = []
    for n in n_factors:
        comb_factors = [elem for elem in list(combinations(l, n))]
        divisors = list(map(producte, comb_factors))
        ll.append(divisors)
    div = list(set([e for elem in ll for e in elem])) #el set() és per eliminar repetits i ORDENA
    return div

# optimitza la mida del chunk amb una fita inferior de MB
def fita(param, div, cols):
    for i,elem in enumerate(div):
        dim = ((cols/elem)**2)*8/(2**20) #en MB
        if (dim <= param['fita_chunk_inf']) and (i != 0):
            #les fites estan en dos condicionals diferents pq m'interesa la dimesió del chunk
            #minima que cumpleixi les condicions
            if ((cols/div[i-1])**2)*8/(2**20) <= param['fita_chunk_sup']: #Fita superior.
                return div[i-1]
            else:
                print("- ERROR - Can't find F(x)'s optimal quadratic minors. Try other wavelon config.", '\n', "A number of wavelons divisible by a number in range [", round(np.sqrt(param['fita_chunk_sup']*2**20/8)), ':', round(np.sqrt(param['fita_chunk_inf']*2**20/8)), "] is needed.")
                exit()
    return 1

# num minim de Iapps per poder quadrar les dimensions i tenir un mínim de 1 bloc amb n chunks.
def n_Iapps_minim(div, cols, punts, mida_chunk):
    n_Iapps_min = 1
    while(True): #incrementarà el llindar per obtenir el mínim de Iapps
        for i, elem in enumerate(div):
            if (cols % punts) == 0:
                return n_Iapps_min*cols//punts
            if cols/elem <= n_Iapps_min:
                if i == 0:
                    return cols #si el min és + gran que el nº de wavelons retorno nº wavelons
                if (((cols//div[i-1]*punts) % mida_chunk) == 0) and (cols//div[i-1]*punts > cols):
                    return cols//div[i-1] #Em quedo amb l'anterior n_Iapps, pq m'he passat del min.
        n_Iapps_min += 100

# where magic happens
def chunk_size(param, cols, punts, multiplier):
    div = [1]+divisors(descomp_factorial(cols))
    
    j = fita(param, div, cols)
    mida_chunk = cols//j #mida_chunk són les file si les columnes (menor quadrat)
    
    n_Iapp = n_Iapps_minim(div, cols, punts, mida_chunk)
    
    print('') #fico aquest fake salt de linia perque no sem descuadri la taula
    print(display_configuration(j, cols, multiplier, n_Iapp, punts, mida_chunk))
    #multipler es per aumentar més les Iapps si podem
    answer = input("Want to change multiplier [m] or number of points [p]? (m/p): ")
    if answer == 'm':
        new_multiplier = input('Enter new int value: ')
        multiplier = int(new_multiplier)
        return chunk_size(param, cols, punts, multiplier)
    elif answer == 'p':
        new_points = input('Enter new int value: ')
        punts = int(new_points)
        return chunk_size(param, cols, punts, multiplier)
    print('')
    return multiplier*n_Iapp*punts//mida_chunk, multiplier*n_Iapp, mida_chunk, punts
        
import multiprocessing as mp
def accel(n_blocs, n, wavelons):
    #fico el mp.cpu_count() per tenir-ho preparat per quan vulgui parallelitzar
    div = divisors(descomp_factorial(n_blocs))
    div.reverse() #comença pels divisors més grans
    for elem in div:
        if (n*elem*wavelons)*8/(2**30) <= psutil.virtual_memory().total/(1024**3)/mp.cpu_count():
    	    return elem
    return 1

############################################ GRAPHICS ############################################

##Taula de la terminal
from terminaltables import SingleTable

def display_configuration(j, cols, multiplier, n_Iapp, punts, mida_chunk):
    config = [['n_chunks', j*multiplier*n_Iapp*punts//mida_chunk, 'n_points', punts],
             ['n_blocs', multiplier*n_Iapp*punts//mida_chunk, 'n_Iapps', multiplier*n_Iapp],
             ['files_chunk', mida_chunk],
             ['Memoria F(x)', str(round(punts*(multiplier*n_Iapp)*cols*8/(2**30), 2)) + ' GB', 'Memoria/chunk', str(round((mida_chunk**2)*8/(2**20),2)) + ' MB']
    ]
    taula = SingleTable(config, ' WAVENET WITH '+str(cols)+' WAVELONS ')
    taula.inner_row_border = True
    taula.justify_columns = {0: 'center', 1: 'center', 2: 'center', 3: 'center'}
    return taula.table

### GRAPHICS
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

#graphic's format
pylab.rcParams['figure.figsize'] = 11, 12
pylab.rcParams['axes.titlesize'] = 20
pylab.rcParams['axes.labelsize'] = 16
pylab.rcParams['xtick.labelsize'] = 14
pylab.rcParams['ytick.labelsize'] = 14
pylab.rcParams['legend.fontsize'] = 14

def phase_portrait(out_1, out_2, out_3, predict_1, predict_2, predict_3, var, titol):
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title(titol)
    ax.plot(out_1, out_3, out_2, label='Target', color='blue', linestyle='-', lw = 0.2)
    ax.plot(predict_1, predict_3, predict_2, label='WNN', color='orange', linestyle='-', lw = 0.1)
    ax.set_xlabel(var[0])
    ax.set_ylabel(var[2])
    ax.set_zlabel(var[1])
    ax.legend()    
    plt.show()

def time_graphic(target, Iapps, predicted_data, nom_ordenades, titol):
    
    time=[]
    [time.append(i) for i in range(len(target))]
    
    plt.figure()
    plt.subplot(211)
    plt.title(titol)
    plt.xlabel('Steps')
    plt.ylabel(nom_ordenades)
    plt.plot(time, target, label='Target', color='blue', linestyle='-')
    plt.plot(time, predicted_data, label='WNN', color='orange', linestyle='-')
    plt.legend()

    plt.subplot(212)
    plt.title('Randomised forced term')
    plt.xlabel('Steps')
    plt.ylabel('Iapp')
    plt.plot(time, Iapps, color='blue', marker=',', linestyle='-')
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.10, right=0.95, hspace=0.5, wspace=0.35)
    plt.show()
