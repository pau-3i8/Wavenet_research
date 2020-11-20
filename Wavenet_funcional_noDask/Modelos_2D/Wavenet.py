from hyperlearn.numba import _sum, eigh
from hyperlearn.random import uniform
from hyperlearn.solvers import solve
from itertools import product, groupby
from scale_func import phi, psi
from numpy.linalg import lstsq
from tqdm import tqdm
import numpy as np

########################### MATRIZ FUNCIONES DE ACTIVACIÓN 3 ENTRADAS #############################

def matriu_Fx(param, input_1, input_2, Iapp):
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
        matriu[0] = (np.ones((1, N))*input_1)
        matriu[1] = (np.ones((1, N))*input_2)
    
    ## Creas las columnas de funciones de escala
    if param['bool_scale']:
        n = [n for n in range(n_sf)]
        for ns in tqdm(list(product(n,n,n)), desc='Computing scale functions', unit='column', leave=False):
            n1, n2, n3 = ns
            matriu[i] = phi(sf_name, input_1, n1)* phi(sf_name, input_2, n2)* phi(sf_name, Iapp, n3)
            i+=1

    ## Creas las columnas de wavelets
    for m in tqdm(range(param['resolution']+1), desc='Computing wavelets', unit='level', leave=False):
        n = [n for n in range(n_sf)]
        c = [c for c in range(2**m)]
        v = [(input_1, input_2, Iapp), (Iapp, input_1, input_2), (input_2, Iapp, input_1)]

        for elem in list(product(n,n,n)):
            n1, n2, n3 = elem
            for var in v:
                for c1 in c:
                    matriu[i] = phi(sf_name, var[0], n1)* phi(sf_name, var[1], n2)* psi(sf_name, (2**m)* var[2] - c1, n3)
                    i+=1
                for ci in list(product(c,c)):
                    c1, c2 = ci
                    matriu[i] = phi(sf_name, var[0], n1)* psi(sf_name, (2**m)* var[1] - c1, n2)* psi(sf_name, (2**m)* var[2] - c2, n3)
                    i+=1
            for ci in list(product(c,c,c)):
                c1, c2, c3 = ci
                matriu[i] = psi(sf_name, (2**m)* input_1 - c1, n1)* psi(sf_name, (2**m)* input_2 - c2, n2)* psi(sf_name, (2**m)* Iapp - c3, n3)
                i+=1
                
    return matriu.T

################################## FUNCIONES PARA GENERAR DATOS ###################################

### CREATE IAPPS
def generate_data(param, n_Iapps):
    Iapp = uniform(param['I_min'], param['I_max'], n_Iapps)
    Iapp = [key for key, group in groupby(Iapp) for n_times in range(param['points'])]
    return np.array(Iapp, dtype='float32')

### IMPORT IAPPS FROM A FILE
def import_data(param):
    Iapp = read_data('inputs.txt')
    Iapp = [key for key, group in groupby(Iapp) for n_times in range(param['points'])]
    return np.array(Iapp, dtype='float32')

### CREATE A POSTPROCESSED GRID OF POINTS WITH THE IAPPS
def posprocessed_data(param, train_data, Iapps):
    ## Save the domain limits to normalize the data within this limits
    param['max_1'] = 0.00023
    param['min_1'] = -0.2355
    param['max_2'] = 1.0535
    param['min_2'] = -0.3235
    ## Normalitze inputs
    input_1, input_2, Iapps = normalize(param, train_data[0], train_data[1], Iapps)
    return input_1, input_2, Iapps

#################################### FUNCTIONES DEL ALGORITMO #####################################

### LOSS
def loss(wavelet, target):
    ## Fastest
    return np.array([solve(wavelet, target, alpha=1e-6)])
    """
    ## Second fastest
    FTF = (wavelet.T).dot(wavelet)
    vaps, _ = eigh(FTF)
    mu = 1e-18
    A = FTF + np.identity(FTF.shape[1])*mu*vaps[-1].real
    b = (wavelet.T).dot(target)
    return lstsq(A, b, rcond=None)[0].T
    """
    
### OBTEIN WEIGHTS
def training(param, Fx, target, var):
    print('--- Aproximando', var, '---')
    weights = loss(Fx, target)
    ## Compute approximation error
    print('->', var, 'MSE at level =', param['resolution']+2, 'is:', _sum((target-Fx.dot(weights.T))**2)/len(target))
    return weights.T
    
### NORMALITZATION
def normalize(param, input_1, input_2, I):
    norm1 = (input_1-param['min_1'])/(param['max_1']-param['min_1'])
    norm2 = (input_2-param['min_2'])/(param['max_2']-param['min_2'])
    Inorm = (I-param['I_min'])/(param['I_max']-param['I_min'])
    return norm1, norm2, Inorm

############################################ DOCUMENTOS ###########################################

## Serveix per guardar Iapps, pesos i outputs pq els vectors tenen la mateixa dimensió
def save_data(arxiu, vector):
    f = open(arxiu, 'w')
    [f.writelines([str(vector[i])+'\n']) for i in range(len(vector))]
    f.close()
    
## Serveix per llegir Iapps
def read_data(arxiu):
    f = open(arxiu, 'r')
    data = [float(line.split()[0]) for line in f]
    f.close()
    return data

## Serveix per llegir pesos
def read_weights(arxiu):
    f = open(arxiu, 'r')
    weights = [[float(line.replace('[', '').replace(']', ''))] for line in f]
    f.close()
    return weights
