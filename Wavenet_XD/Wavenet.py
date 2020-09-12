from scale_func import select_phi_scaled, select_psi
from hyperlearn.numba import _sum, mean, eigh
from hyperlearn.random import uniform
from hyperlearn.solvers import solve
from itertools import product, groupby
from numpy.linalg import lstsq
from numba import njit
import numpy as np
import random

################################# MATRIZ FUNCIONES DE ACTIVACIÓN ##################################

def lineal(input_data):
    lineal = (np.ones((1, len(input_data)))*input_data).T
    return lineal

def scale(sf_name, input_data, n_sf, lineal, bool_lineal):
    wavelets = np.zeros((len(input_data), n_sf)).T
    i = 0
    for n1 in range(n_sf):
        wavelets[i] = select_phi_scaled(sf_name, input_data, n1)
        i += 1
    wavelets = wavelets.T
    if bool_lineal:
        wavelets = np.append(lineal, wavelets, axis=1)
    return wavelets

def wavelets(sf_name, input_data, m, wavelets, n_sf):
    N = len(input_data)
    aux = np.zeros((N, n_sf*(2**m))).T
    i = 0
    for n1 in range(n_sf):
        for n in range(2**m):
            aux[i] = select_psi(sf_name, (2**m)* input_data - n, n1)
            i+=1
    wavelets = np.append(wavelets, aux.T, axis=1)
    return wavelets

## Construeixo una matriu per cada input (pq els resolc per separat)
def matriu_Fx(param, input_data):
    ## Creas las columnas de la parte lineal
    wavelet = lineal(input_data)
    ## Creas las columnas de funciones de escala
    if param['bool_scale']:
        wavelet = scale(param['fscale'], input_data, param['n_sf'], wavelet, param['bool_lineal'])
    ## Creas las columnas de wavelets
    for m in range(param['resolution']+1):
        wavelet = wavelets(param['fscale'], input_data, m, wavelet, param['n_sf'])
    return wavelet

################################## FUNCIONES PARA GENERAR DATOS ###################################

### CREATE IAPPS
def generate_data(param, n_Iapps):
    Iapp = uniform(param['I_min'], param['I_max'], n_Iapps)
    Iapp = [key for key, group in groupby(Iapp) for n_times in range(param['points'])]
    return np.array(Iapp)

### IMPORT IAPPS FROM A FILE
def import_data(param):
    Iapp = read_data('inputs.txt')
    Iapp = [key for key, group in groupby(Iapp) for n_times in range(param['points'])]
    return np.array(Iapp)

### CREATE A POSTPROCESSED GRID OF POINTS WITH THE IAPPS
def posprocessed_data(param, all_data):
    target, input_data = all_data
    ## Adjust the dimension of the target vector for the WN
    redimensioned_target = np.array([[elem] for elem in list(target)])
    ## Save the domain limits to normalize the data within this limits
    param['max'] = np.max(input_data)
    param['min'] = np.min(input_data)
    param['target'] = target
    ## Normalitze inputs
    input_data = normalize(param, input_data)
    return redimensioned_target, input_data

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
def training(param, wavelet, target):
    weights = loss(wavelet, target)
    ## Compute approximation error
    print('MSE at level = ' + str(param['resolution']+2) + ' is: ' + str(_sum((target - np.dot(wavelet, weights.T))**2)/len(target)))
    return weights.T

### PREDICTION
def prediction(param, weights, input_data):
    predicted_data = (matriu_Fx(param, input_data).dot(weights)).T[0]
    ## Compute prediction's error
    target = param['target']
    print('Prediction RMSE: ' + str((np.sum((target-predicted_data)**2)/np.sum((np.ones((1,len(target)))*mean(target)-target)**2))*100) + '%')
    return predicted_data
    
### NORMALITZATION
def normalize(param, input_data):
    return (input_data-param['min'])/(param['max']-param['min'])

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
