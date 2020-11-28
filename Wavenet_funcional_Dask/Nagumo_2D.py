from dask.distributed import Client, performance_report, progress
from itertools import product, combinations, groupby
import dask.dataframe as dd
import dask.array as da
from tqdm import tqdm
import pandas as pd
import numpy as np
import Wavenet
import random
import psutil
import h5py

#Per monitorar el scheduler: http://localhost:8787

#Cluster local
client = Client(processes=False, silence_logs='error')

############################################## EULER ##############################################

### EULER
def euler(param, Iapp, w0, v0):
    ##Variables for saving the inputs
    v_train = np.zeros_like(Iapp)
    w_train = np.zeros_like(Iapp)
    #Variables for saving the targets
    target_v = np.zeros_like(Iapp)
    target_w = np.zeros_like(Iapp)
    #Parameters
    h = param['h']
    a = param['a']
    eps = param['eps']
    gamma = param['gamma']
    n_points = param['points']
    
    Is = [key for key, group in groupby(Iapp)]
    i=0
    for I in tqdm(Is, desc='Generant dades', unit='integracions', leave=False):
        wn = np.zeros(n_points+1)
        vn = np.zeros(n_points+1)
        wn[0] = w0
        vn[0] = v0
        for k in range(n_points):#podries ficar un trange però aquest loop es molt rapid i no cal
            v1 = v0 + h * (-v0*(v0-1)*(v0-a)+w0+I) #dv/dt = -f(v)+w+Ibase
            w1 = w0 + h * eps*(-v0-gamma*w0) #dw/dt = eps*(-v-gamma*w)
            wn[k+1] = w1
            vn[k+1] = v1
            v0=v1
            w0=w1
        for j in range(n_points):
            target_w[i+j] = wn[1:][j]#n+1
            target_v[i+j] = vn[1:][j]
            w_train[i+j] = wn[:-1][j]#n
            v_train[i+j] = vn[:-1][j]
        i+=n_points
    return (w_train, v_train), (target_w, target_v)

################################### PARAMS & WAVENET PRECONFIG. ###################################

param0 = {'a':0.14, 'eps':0.01, 'gamma':2.54, 'h':0.1, 'points':1500, 
          'n_Iapp':None, 'n_Iapp_new':1, 'I_max':0.1, 'I_min':0.0,
          'resolution':1, 'n_sf':5, 'rechunk_factor':1,
          'fscale':'bicubic', 'bool_lineal':True, 'bool_scale':True,
          'generateIapp':True, 'generateNewIapp':True, 'shuffle':False}

param0['fita_chunk_inf'] = 130 #[MB]
param0['fita_chunk_sup'] = 175 #[MB]

CI_approx = {'w0':-0.1, 'v0':0.0}
CI_simu = {'w0':-0.1, 'v0':0.0}

outputs = ['W_predict.txt', 'V_predict.txt']
pesos = ['weights_w.parquet', 'weights_v.parquet']
var = ['w', 'v', 'Iapp'] #fico Iapp pq reaprofito les funcions pels grafics i necessito var[2]

##################################### Execució de l'algorisme #####################################

def approximation(param, input_1, input_2, Iapps, target_data, f_c, n_blocs, wavelons):
    
    a = Wavenet.accel(n_blocs, f_c, wavelons)
    n = f_c*a
    n_blocs //= a
    with h5py.File('matriu.hdf5', 'w') as f:
        dset = f.create_dataset('dataset_matriu', shape = (len(Iapps), wavelons),
                                chunks = (f_c, f_c), compression = 'gzip', 
                                compression_opts = 6, dtype = 'float64')
        for i in tqdm(range(n_blocs), desc='Guardant matriu', unit='chunk', leave=False):
            dset[n*i:n*(i+1)] =  Wavenet.matriu_Fx_2D(param,
                                                      input_1[n*i:n*(i+1)],
                                                      input_2[n*i:n*(i+1)], 
                                                      Iapps[n*i:n*(i+1)])
    
    with performance_report(): #genera dask-report.html
        f = h5py.File('matriu.hdf5', 'r')
        FX = da.from_array(f.get('dataset_matriu'), chunks=(int(param['rechunk_factor'])*f_c, f_c))
        print('Guardant matriu de covariancia FTF')
        FTF = dd.from_dask_array(da.dot(FX.T, FX), columns = [str(elem) for elem in np.arange(wavelons)])
        FTF = FTF.persist() #he d'executar els calculs al background per veure el progres
        progress(FTF) #per veure la barra de progress amb distributed.
        FTF.to_parquet('FTF.parquet')
        print('')
        for index in range(len(target_data)):
            print('--- Aproximando', var[index], '---')
            target = np.array([[elem] for elem in target_data[index]])
            Wavenet.training(param['resolution'], FX, target, f_c, var[index])
        f.close()
    print('')
    
def simulation(param, ci_1, ci_2, target, Iapps):
    weights_1 = np.array(pd.read_parquet(pesos[0]))
    weights_2 = np.array(pd.read_parquet(pesos[1]))
    t_1, t_2 = target[0], target[1] #targets
    i_1, i_2 = np.array([ci_1]), np.array([ci_2]) #inputs
    predicted_1 = np.zeros_like(target[0])
    predicted_2 = np.zeros_like(target[1])
    for j,I in enumerate(tqdm(Iapps, desc='Predicció', unit=' integracions', leave=False)):
        i_1, i_2, I = normalize(param, (i_1, i_2), np.array([I]))
        Fx = Wavenet.matriu_Fx_2D(param, i_1, i_2, I)
        i_1 = (Fx.dot(weights_1))[0] #wn+1
        i_2 = (Fx.dot(weights_2))[0] #vn+1
        predicted_1[j] = i_1
        predicted_2[j] = i_2
    print('->', var[0],'RMSE:', np.sum((t_1-predicted_1)**2)/np.sum((np.mean(t_1)-t_1)**2)*100,'%')
    print('->', var[1],'RMSE:', np.sum((t_2-predicted_2)**2)/np.sum((np.mean(t_2)-t_2)**2)*100,'%')
    Wavenet.save_data(outputs[0], predicted_1)
    Wavenet.save_data(outputs[1], predicted_2)
    
### NORMALITZATION
def normalize(param, train_data, I):
    param['max_1'] = 0.00023 #np.max(train_data[0])
    param['min_1'] = -0.2355 #np.min(train_data[0])
    param['max_2'] = 1.0535  #np.max(train_data[1])
    param['min_2'] = -0.3235 #np.min(train_data[1])
    norm1 = (train_data[0]-param['min_1'])/(param['max_1']-param['min_1'])
    norm2 = (train_data[1]-param['min_2'])/(param['max_2']-param['min_2'])
    Inorm = (I-param['I_min'])/(param['I_max']-param['I_min'])
    return norm1, norm2, Inorm

################################# Funció principal de l'algorisme #################################

def Nagumo_2D():
    param = dict(param0)
    ## Memòria que es farà servir:
    if param['bool_lineal']: l = 2
    else: l = 0
    if param['bool_scale']: neurones = 1
    else: neurones = 0
    for m in range(param['resolution']+1):
        neurones+=(2**(3*m)+3*2**(2*m)+3*2**m)
    wavelons = l+(param['n_sf']**3)*neurones
    n_blocs, param['n_Iapp'], f_chunk, param['points'] = Wavenet.chunk_size(param, wavelons, param['points'], 1)
    memoria = param['points']*param['n_Iapp']*wavelons*8/(2**30)
    compressio = 0.2 #essent conservadors (limit=17%)
    if memoria <= psutil.disk_usage('/')[2]/(1024**3)/compressio:
        pass
    else:
        print('\n','- ERROR - No hay suficiente memoria para la configuración de la Wavenet.')
        exit()
    
    ## Aquí aproximo la funció
    if param['generateIapp']: Iapps = Wavenet.generate_data(param, param['n_Iapp'])
    else: Iapps = Wavenet.import_data(param)
    Wavenet.save_data('inputs.txt', Iapps)
    all_data = euler(param, Iapps, CI_approx['w0'], CI_approx['v0'])
    train_data, target_data = all_data
    input_1, input_2, Iapps = normalize(param, train_data, Iapps)
    #dim_bloc = (files_chunk -> f_chunk, wavelons)
    approximation(param, input_1, input_2, Iapps, target_data, f_chunk, n_blocs, wavelons)
    
    ## Aquí predic el comportament
    param['points'] = 1000 #simularé menys dels que aproximo per no morir de vell.
    if param['generateNewIapp']: Iapps = Wavenet.generate_data(param, param['n_Iapp_new'])
    else:
        Iapps = [Iapp for Iapp, group in groupby(Wavenet.import_data(param))][:param['n_Iapp_new']]
        if param['shuffle']: random.shuffle(Iapps)
        Iapps = [Iapp for Iapp in Iapps for n_times in range(param['points'])]
    _, target = euler(param, Iapps, CI_simu['w0'], CI_simu['v0'])
    simulation(param, CI_simu['w0'], CI_simu['v0'], target, Iapps)
    
    ## Gràfics
    w, v = target
    visual(w, v, Iapps, 'FitzHugh-Nagumo 2D outputs')
    
##Gràfics dels resultats
def visual(target_1, target_2, Iapps, titol):
    with open(outputs[0], 'r') as f1:
        predict_1 = [float(line.replace('[', '').replace(']', '')) for line in f1]
    with open(outputs[1], 'r') as f2:
        predict_2 = [float(line.replace('[', '').replace(']', '')) for line in f2]
        
    Wavenet.phase_portrait(target_1, target_2, Iapps, predict_1, predict_2, Iapps, var, titol)
    Wavenet.time_graphic(target_1, Iapps, predict_1, var[0], titol)
    Wavenet.time_graphic(target_2, Iapps, predict_2, var[1], titol)
    
if __name__ == "__main__":
    from timeit import Timer
    print('Execution time:', Timer(lambda: Nagumo_2D()).timeit(number=1), 's')
