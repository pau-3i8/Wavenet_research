from dask.diagnostics import ProgressBar, Profiler, ResourceProfiler, CacheProfiler, visualize
from dask.distributed import Client, performance_report, progress
from itertools import product, groupby
from tqdm.auto import trange
from decimal import Decimal
import dask.dataframe as dd
import dask.array as da
from tqdm import tqdm
import pandas as pd
import numpy as np
import Wavenet
import random
import psutil
import h5py
import dask

#client = Client(processes=False)

### EULER
def euler(param, Iapp):
    ##Variables for saving the inputs
    v_train = np.zeros_like(Iapp)
    w_train = np.zeros_like(Iapp)
    v_train2 = np.zeros_like(Iapp)
    w_train2 = np.zeros_like(Iapp)
    #Variables for saving the targets
    target_v = np.zeros_like(Iapp)
    target_w = np.zeros_like(Iapp)
    #Parameters
    w0, v0 = param['w0'], param['v0'] #initial cond.
    h = param['h']
    a = param['a']
    eps = param['eps']
    beta = param['beta']
    n_points = param['points']
    
    Is = [key for key, group in groupby(Iapp)]
    i=0
    for I in tqdm(Is, desc='Generant dades', unit='integracions', leave=False):
        wn = np.zeros(n_points+2)
        vn = np.zeros(n_points+2)
        wn[0] = w0
        vn[0] = v0
        for k in range(n_points+1):#podries ficar un trange però aquest loop es molt rapid i no cal
            v1 = v0 + h * (-v0*(v0-1)*(v0-a)+w0+I) #dv/dt = -f(v)+w+Ibase
            w1 = w0 + h * (eps*v0+beta*w0) #dw/dt = eps*(-v-gamma*w)
            wn[k+1] = w1
            vn[k+1] = v1
            v0=v1
            w0=w1
        for j in range(n_points):
            target_w[i+j] = wn[2:][j]#n+2
            target_v[i+j] = vn[2:][j]
            w_train2[i+j] = wn[1:-1][j]#n+1
            v_train2[i+j] = vn[1:-1][j]
            w_train[i+j] = wn[:-2][j]#n
            v_train[i+j] = vn[:-2][j]
        i+=n_points
    param['w2']=w_train2[0]
    param['v2']=v_train2[0]
    return ((w_train, w_train2), (v_train, v_train2)), (target_w, target_v)

#feq. mínima pel Nagumo 2D = 1500 punts

### PARAMS & WAVENET CONFIG.
param0 = {'a':0.14, 'eps':-0.01, 'beta':-0.01*2.54, 'h':0.1, 'points':2126,
          'n_Iapp':None, 'n_Iapp_new':1, 'I_max':0.1, 'I_min':0, 'resolution':1, 'n_sf':5,
          'fscale':'bicubic', 'bool_lineal':True, 'bool_scale':True,
          'generateIapp':False, 'generateNewIapp':True, 'shuffle':False}

outputs = ['W_predict.txt', 'V_predict.txt']
pesos = ['weights_w.txt', 'weights_v.txt']
var = ['w', 'v']

### CREO LA MIDA DE CHUNKS
#si vols n_Iapp = n_chunks caldrà que fixis el número de punts = número de wavelons
def chunk_size(cols, punts, multiplier):
    num = cols/punts
    if num == 1: num = int(num)
    N = 10**abs(Decimal(str(num)).as_tuple().exponent)
    print('--------- Wavenet configuration ---------')
    print('wavelons:', cols, '||', 'n_points:', punts)
    print('n_chunks:', multiplier*N, '||', 'n_Iapps:', multiplier*N*cols//punts)
    memoria = punts*(multiplier*N*cols//punts)*cols*8/(2**30)
    print('Memoria necesaria:', round(memoria, 2), 'GB aprox.')
    print('-----------------------------------------')
    #multipler es per aumentar més les Iapps si podem
    answer = input("Want to change multiplier [m] or number of points [p]? (m/p): ")
    print('')
    if answer == 'm':
        new_multiplier = input('Enter new int value: ')
        multiplier = int(new_multiplier)
        return chunk_size(cols, punts, multiplier)
    elif answer == 'p':
        new_points = input('Enter new int value: ')
        punts = int(new_points)
        return chunk_size(cols, punts, multiplier)
    return multiplier*N, multiplier*N*cols//punts, punts
    
### OBTAIN WEIGHTS
def approximation(param, all_data, Iapps, mida_chunk, n_chunks):
    train_data, target_data = all_data
    train_data  = (train_data[0][0], train_data[1][0])
    input_1, input_2, Iapps = Wavenet.posprocessed_data(param, train_data, Iapps)
    
    with h5py.File('matriu.hdf5', 'w') as f:
        #guarda la matriu per chunks
        for i in tqdm(range(n_chunks), desc='Guardant matriu', unit='chunk', leave=False):
            norm_input_1 = input_1[mida_chunk*i:mida_chunk*(i+1)]
            norm_input_2 = input_2[mida_chunk*i:mida_chunk*(i+1)]
            norm_Iapps = Iapps[mida_chunk*i:mida_chunk*(i+1)]
            chunk = Wavenet.matriu_Fx(param, norm_input_1, norm_input_2, norm_Iapps)
            f.create_dataset('chunk'+'0'*(16-len(str(i)))+str(i), data = chunk,
                             compression = 'gzip', compression_opts = 6, dtype = 'float64')
    #el with ja em tanca l'arxiu.
    
    ProgressBar().register() #només mesurarà el codi que faci servir dask.
    #.register() per activar diagnostics globals
    #Si fas servir Profiler().register() caldrà que després netegis el resultats antics (.clear())
    with Profiler() as prof, ResourceProfiler(dt=0.25) as rprof, CacheProfiler() as cprof:
        f = h5py.File('matriu.hdf5', 'r')
        datasets = []
        for key in tqdm(list(f.keys()), desc='Important matriu', unit='chunk', leave=False):
            chunk = f.get(key) #un menor es un dataset
            x = da.from_array(chunk, chunks = mida_chunk)
            datasets.append(x)
        FX = da.concatenate(datasets, axis=0)
        print('Guardant matriu de covariancia FTF')
        dd.from_dask_array(da.dot(FX.T, FX), columns = [str(elem) for elem in np.arange(mida_chunk)]).to_parquet('FTF.parquet')
        for index in range(len(train_data)):
            print('Solving '+var[index])
            target = np.array([[elem] for elem in target_data[index]])
            #guarda pesos en un arxiu parquet
            Wavenet.training(param['resolution'], FX, target, mida_chunk, var[index])
        f.close()
    visualize([prof, rprof, cprof])
    
### EXECUTE THE SIMULATION FROM A FILE WITH INPUTS
def simulation(param, target, Iapps):
    weights_w = np.array(pd.read_parquet('weights_w.parquet'))
    weights_v = np.array(pd.read_parquet('weights_v.parquet'))
    target_w = target[0]
    target_v = target[1]
    input_1 = np.array([param['w0']])
    input_2 = np.array([param['v0']])
    predicted_1 = np.zeros_like(target[0])
    predicted_2 = np.zeros_like(target[1])
    for j,I in enumerate(tqdm(Iapps, desc='Predicció completada', unit='integracions', leave=False)):
        #reaprofito el max i el min de les dades entrenades
        """
        if not (param['min_'+str(index+1)] <= input_1 <= param['max_'+str(index+1)] and param['min_'+str(index+2)] <= input_2 <= param['max_'+str(index+2)]):#per donar marge fico el x10
            print('Parat a la integració', j)
            break
        """
        norm_input_1, norm_input_2, I = Wavenet.normalize(param, input_1, input_2, np.array([I]))
        vector_Fx = Wavenet.matriu_Fx(param, norm_input_1, norm_input_2, I)
        ##Aquí Fx només és un vector, perquè treballo amb un punt cada vegada
        input_1 = (vector_Fx.dot(np.array(weights_w)))[0]
        input_2 = (vector_Fx.dot(np.array(weights_v)))[0]
        predicted_1[j] = input_1
        predicted_2[j] = input_2
    print('-> w RMSE:', np.sum((target_w-predicted_1)**2)/np.sum((np.mean(target_w)-target_w)**2)*100,'%')
    print('-> v RMSE:', np.sum((target_v-predicted_2)**2)/np.sum((np.mean(target_v)-target_v)**2)*100,'%')
    Wavenet.save_data(outputs[0], predicted_1)
    Wavenet.save_data(outputs[1], predicted_2)
    
def Nagumo_2D():
    param = dict(param0)
    
    ## Memòria que es farà servir:
    if param['bool_lineal']: l = 2
    if param['bool_scale']: neurones = 1
    else: neurones = 0
    for m in range(param['resolution']+1):
        neurones+=(2**(3*m)+3*2**(2*m)+3*2**m)
    wavelons = l+(param['n_sf']**3)*neurones
    n_chunks, param['n_Iapp'], param['points'] = chunk_size(wavelons, param['points'], 1)
    memoria = param['points']*param['n_Iapp']*wavelons*8/(2**30)
    compressio = 0.2 #essent conservadors (limit=17%)
    if memoria <= psutil.disk_usage('/')[2]/(1024**3)/compressio:
        pass
    else:
        print('No hay suficiente memoria para la configuración de la Wavenet')
        exit()

     ## Aquí aproximo la funció
    if param['generateIapp']: Iapps = Wavenet.generate_data(param, param['n_Iapp'])
    else: Iapps = Wavenet.import_data(param)
    param['Iapp'] = Iapps
    ## Creating the grid of points with the all_data variable
    param['w0'] = -0.1
    param['v0'] = 0.0
    Wavenet.save_data('inputs.txt', param['Iapp']) #guardes les Iapps de l'entrenament
    all_data = euler(param, param['Iapp'])
    #wavelons = n_columnes = mida_chunk (els chunks són matrius quadrades)
    approximation(param, all_data, param['Iapp'], wavelons, n_chunks)
    
    param['points'] = 1500
    ## Aquí predic el comportament
    if param['generateNewIapp']: Iapps = Wavenet.generate_data(param, param['n_Iapp_new'])
    else:
        Iapps = [Iapp for Iapp, group in groupby(param['Iapp'])]
        if param['shuffle']: random.shuffle(Iapps)
        Iapps = [Iapp for Iapp in Iapps for n_times in range(param['points'])]
    param['Iapp'] = Iapps
    param['w0'] = -0.1
    param['v0'] = 0.0
    _, target = euler(param, param['Iapp'])
    simulation(param, target, param['Iapp'])
    
    ## Gràfics
    w, v = target
    visual(param, w, v, param['Iapp'])
    
def visual(param, w_target, v_target, Iapps):
    f1 = open('W_predict.txt', 'r')
    f2 = open('V_predict.txt', 'r')
    w_predict = [float(line.replace('[', '').replace(']', '')) for line in f1]
    v_predict = [float(line.replace('[', '').replace(']', '')) for line in f2]
    f1.close()
    f2.close()
    phase_portrait(w_target, v_target, Iapps, w_predict, v_predict)
    time_graphic(w_target, param['Iapp'], w_predict, 'w output')
    time_graphic(v_target, param['Iapp'], v_predict, 'v output')

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

def phase_portrait(wout, vout, Iapps, w_predict, v_predict):
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title('FitzHugh-Nagumo 2D')
    ax.plot(w_predict, Iapps, v_predict, label='WNN', color='orange', marker=',', linestyle='')
    ax.plot(wout, Iapps, vout, label='Target', color='blue', marker=',', linestyle='')
    ax.set_xlabel('w')
    ax.set_ylabel('Iapp')
    ax.set_zlabel('v')
    ax.legend()
    plt.show()

def time_graphic(target, Iapp, predicted_data, nom_ordenades):
    
    time=[]
    [time.append(i) for i in range(len(target))]
    
    plt.figure()
    plt.subplot(211)
    plt.title('FitzHugh-Nagumo')
    plt.xlabel('Steps')
    plt.ylabel(nom_ordenades)
    plt.plot(time, target, label='Target', color='blue', marker=',', linestyle='-')
    plt.plot(time, predicted_data, label='WNN', color='orange', marker=',', linestyle='-')
    plt.legend()

    plt.subplot(212)
    plt.title('Randomised forced term')
    plt.xlabel('Steps')
    plt.ylabel('Iapp')
    plt.plot(time, Iapp, color='blue', marker=',', linestyle='-')
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.10, right=0.95, hspace=0.5, wspace=0.35)
    plt.show()
        
if __name__ == "__main__":
    from timeit import Timer
    print('Execution time:', Timer(lambda: Nagumo_2D()).timeit(number=1), 's')
