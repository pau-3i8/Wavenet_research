from itertools import product, groupby
from activation_funcs import phi, psi
import dask.dataframe as dd
import dask.array as da
from tqdm import tqdm
import pandas as pd
import numpy as np
import h5py

########################### MATRIZ FUNCIONES DE ACTIVACIÓN 3 ENTRADAS #############################
## Construeixo una matriu per cada input (pq els resolc per separat)
def matriu_Fx(param, input_1, input_2, Iapp):
    sf_name = param['fscale']
    n_sf = param['n_sf']
    N = len(input_1)
    
    ## Creas las columnas de la parte lineal
    if param['bool_lineal']:
        lineal_1 = (np.ones((1, N))*input_1).T
        lineal_2 = (np.ones((1, N))*input_2).T
        matriu = np.append(lineal_1, lineal_2, axis=1)
    
    ## Creas las columnas de funciones de escala
    if param['bool_scale']:
        escala = np.zeros((N, n_sf**3)).T
        n = [n for n in range(n_sf)]
        for i,ns in enumerate(tqdm(list(product(n,n,n)), desc='Computing scale functions', unit='column', leave=False)):
            n1, n2, n3 = ns
            escala[i] = phi(sf_name, input_1, n1)* phi(sf_name, input_2, n2)* phi(sf_name, Iapp, n3)
            i+=1
        if param['bool_lineal']: matriu = np.append(matriu, escala.T, axis=1)
        else: matriu = escala.T

    ## Creas las columnas de wavelets
    for m in tqdm(range(param['resolution']+1), desc='Computing wavelets', unit='level', leave=False):
        n = [n for n in range(n_sf)]
        c = [c for c in range(2**m)]
        v = [(input_1, input_2, Iapp), (Iapp, input_1, input_2), (input_2, Iapp, input_1)]

        aux = np.zeros((N, (n_sf**3)*(2**(3*m)+3*2**(2*m)+3*2**m))).T
        i=0
        for elem in list(product(n,n,n)):
            n1, n2, n3 = elem
            for var in v:
                for c1 in c:
                    aux[i] = phi(sf_name, var[0], n1)* phi(sf_name, var[1], n2)* psi(sf_name, (2**m)* var[2] - c1, n3)
                    i+=1
                for ci in list(product(c,c)):
                    c1, c2 = ci
                    aux[i] = phi(sf_name, var[0], n1)* psi(sf_name, (2**m)* var[1] - c1, n2)* psi(sf_name, (2**m)* var[2] - c2, n3)
                    i+=1
            for ci in list(product(c,c,c)):
                c1, c2, c3 = ci
                aux[i] = psi(sf_name, (2**m)* input_1 - c1, n1)* psi(sf_name, (2**m)* input_2 - c2, n2)* psi(sf_name, (2**m)* Iapp - c3, n3)
                i+=1
        matriu = np.append(matriu, aux.T, axis=1)
    return matriu

################################## FUNCIONES PARA GENERAR DATOS ###################################

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

### CREATE A POSTPROCESSED GRID OF POINTS WITH THE IAPPS
def posprocessed_data(param, train_data, Iapps):
    ## Save the domain limits to normalize the data within this limits
    param['max_1'] = np.max(train_data[0])
    param['min_1'] = np.min(train_data[0])
    param['max_2'] = np.max(train_data[1])
    param['min_2'] = np.min(train_data[1])
    ## Normalitze inputs
    input_1, input_2, Iapps = normalize(param, train_data[0], train_data[1], Iapps)
    return input_1, input_2, Iapps

#################################### FUNCTIONES DEL ALGORITMO #####################################

from numpy.linalg import lstsq, eigh

### LOSS
def training(m, FX, target, mida_chunk, var):
    Y = da.from_array(target, chunks = (mida_chunk, 1))
    df_Y = dd.from_dask_array(da.dot(FX.T, Y), columns = ['0'])
    df_Y.to_parquet('Y.parquet')
    
    FTF = np.array(pd.read_parquet('FTF.parquet'))
    vaps, _ = eigh(FTF)
    mu = 1e-16
    A = FTF + np.identity(FTF.shape[1])*mu*vaps[-1].real
    b = np.array(pd.read_parquet('Y.parquet'))

    weights = lstsq(A, b, rcond=None)[0] #vector columna
    weights = da.from_array(weights, chunks='auto')
    df = dd.from_dask_array(weights, columns = ['0'])
    df.to_parquet('weights_'+var+'.parquet')
    
    #MSE(m, FX, Y, dd.from_pandas(pd.read_parquet('weights_'+var+'.parquet'), npartitions=1))
    #print('-> '+ var + ' MSE at level =', m+2, 'is:',
    #      MSE(FX, Y, np.array(pd.read_parquet('weights_'+var+'.parquet'))))

def MSE(FX, Y, weights):
    w = da.from_array(weights, chunks ='auto')
    MSE = dd.from_dask_array(da.sum((Y - da.dot(FX, w))**2)/len(Y), columns = ['0'])
    MSE.to_parquet('mse.parquet')
    return np.array(pd.read_parquet('mse.parquet'))[0]

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
