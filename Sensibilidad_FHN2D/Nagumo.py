import multiprocessing as mp
from itertools import product, groupby
from Wavenet import scale_fun, mother_wavelet
import Wavenet
from numba import njit
import numpy as np
import random
from hyperlearn.random import uniform
from hyperlearn.numba import _sum, mean

###CREATE IAPPS
def generate_data(param, n_Iapps):
    #Iapp = np.random.uniform(param['I_min'], param['I_max'], n_Iapps)
    Iapp = uniform(param['I_min'], param['I_max'], n_Iapps)
    Iapp = [key for key, group in groupby(Iapp) for n_times in range(param['points'])]
    return Iapp

###IMPORT IAPPS FROM A FILE
def import_data(param):
    Iapp = Wavenet.read_inputs('input.txt')
    Iapp = [key for key, group in groupby(Iapp) for n_times in range(param['points'])]
    return Iapp

###CREATE GRID OF POINTS WITH THE IAPPS
def training_data(param, Iapps):
    w0, y0 = param['w0'], param['y0'] #initial cond.
    Iapp = np.array(Iapps)
    ##Variables for saving the inputs
    y_train = np.zeros_like(Iapp)
    w_train = np.zeros_like(Iapp)
    ##Variables for saving the targets
    target_y = np.zeros_like(Iapp)
    target_w = np.zeros_like(Iapp)
    
    h = param['h']
    a = param['a']
    alfa = param['alfa']
    beta = param['beta']
    n_points = param['points']
    
    Is = [key for key, group in groupby(Iapp)]
    target_w, target_y, w_train, y_train = Wavenet.euler(np.array(Is), w0, y0, n_points, h, a, alfa, beta, target_w, target_y, w_train, y_train)
        
    ##Adjust the dimension of the target vector for the WN
    tary = np.array([[elem] for elem in list(target_y)])
    tarw = np.array([[elem] for elem in list(target_w)])
    
    ##Save the domain limits to normalize the data within this limits
    param['I_max'] = np.max(Iapp)
    param['I_min'] = np.min(Iapp)
    param['y_max'] = np.max(y_train)
    param['y_min'] = np.min(y_train)
    param['w_max'] = np.max(w_train)
    param['w_min'] = np.min(w_train)
    ##Save the non normalized variables for graphics
    param['Iapp'] = Iapp
    param['target_y'] = target_y
    param['target_w'] = target_w
    
    ##Normalitze inputs
    y_train, w_train, Iapp = Wavenet.normalize(param, y_train, w_train, Iapp)
    
    return tary, tarw, y_train, w_train, Iapp

###TRAINING
def matriu_Fx(param, y_train, w_train, Iapp):
    ###LINEAL COMP. MATRIX
    wavelet = Wavenet.lineal(w_train, y_train)
    
    mult = param['n_sf']**3 #j**E
    Nh = 0
    if param['bool_lineal']:
        Nh += len(wavelet[0])
    if param['bool_scale']:
        Nh += mult
        ###SCALE FUNC. MATRIX
        wavelet = Wavenet.scale(param['fscale'], w_train, y_train, Iapp, param['n_sf'], wavelet, param['bool_lineal'])
    
    for m in range(param['resolution']+1):
        Nh += mult*(3*(2**(m))+3*(2**(2*m))+2**(3*m)) #wavelon's growth
        ###WAVELET'S MATRIX 
        wavelet = Wavenet.wavelets(param['fscale'], w_train, y_train, Iapp, m, wavelet, param['n_sf'])

    return Nh, wavelet

def training(param, Nh, wavelet, target):#, return_dic):
    ###LOSS
    weights = Wavenet.loss(wavelet, target, Nh)
    ###COMPUTE ERROR
    print('MSE at level = ' + str(param['resolution']+2) + ' is: ' + str(_sum((target - np.dot(wavelet, weights.T))**2)/len(target)))
    ###RETURN WEIGHTS
    return weights.T
    #return_dic['resultat'] = weights.T

###OBTAIN WEIGHTS
def approximation(param, generate):
    if generate:
        Iapps = generate_data(param, param['n_Iapp'])
    if not generate:
        Iapps = import_data(param)
    tary, tarw, y_train, w_train, Iapp = training_data(param, Iapps)
    
    Nh, wavelet = matriu_Fx(param, y_train, w_train, Iapp)
    
    ##Variables for storing the calculated weights
    weights_y = np.array([weight for weight in training(param, Nh, wavelet, tary)])
    weights_w = np.array([weight for weight in training(param, Nh, wavelet, tarw)])
    
    """
    dict1 = mp.Manager().dict()
    dict2 = mp.Manager().dict()
    proces = []
    proces.append(mp.Process(target = training, args=(param, Nh, wavelet, tary, dict1)))
    proces.append(mp.Process(target = training, args=(param, Nh, wavelet, tarw, dict2)))
    for p in proces:
        p.start()
    for p in proces:
        p.join()
        
    weights_y = dict1.values()[0]
    weights_w = dict2.values()[0]
    """
    
    return weights_y, weights_w

###PREDICTION
def prediction(param, sigma_y, sigma_w, y_train, w_train, Iapp):

    wavelet = Wavenet.lineal(w_train, y_train)
    if param['bool_scale']:
        wavelet = Wavenet.scale(param['fscale'], w_train, y_train, Iapp, param['n_sf'], wavelet, param['bool_lineal'])
    
    for m in range(param['resolution']+1):
        wavelet = Wavenet.wavelets(param['fscale'], w_train, y_train, Iapp, m, wavelet, param['n_sf'])
        
    Y_predict = (np.dot(wavelet, sigma_y)).T[0]
    W_predict = (np.dot(wavelet, sigma_w)).T[0]

    ###SIMULATION'S ERROR
    y = param['target_y']
    print('Prediction RMSE: ' + str((np.sum((y-Y_predict)**2)/np.sum((np.ones((1,len(y)))*mean(y)-y)**2))*100) + '%')
    
    return W_predict, Y_predict

###DISPLAY THE RESULT
def display(param, W_predict, Y_predict):
    answer = input("Show graphic? (y/n): ")
    if answer == 'y':
        Wavenet.graphic(param['target_w'], param['target_y'], param['Iapp'], W_predict, Y_predict)
        Wavenet.graphic_time(param['target_w'], param['target_y'], param['Iapp'], W_predict, Y_predict)
    
###EXECUTE THE SIMULATION FROM A FILE WITH INPUTS
def simulation(param, weights_y, weights_w, generate_new):
    if generate_new:
        Iapps = generate_data(param, param['n_Iapp_new'])
    if not generate_new:
        Iapps = [Iapp for Iapp, group in groupby(param['Iapp'])]
        if param['shuffle']:
            #per desordenar les Iapps (retorna la llista que entra, desordenada)import_data(param)
            random.shuffle(Iapps)
        Iapps = [Iapp for Iapp in Iapps for n_times in range(param['points'])]
        
    _, _, y_train, w_train, Iapp = training_data(param, Iapps)
    
    W_predict, Y_predict = prediction(param, weights_y, weights_w, y_train, w_train, Iapp)
    return W_predict, Y_predict

### EXECUTE WAVENET
param0 = {'a':0.14, 'alfa':-0.01, 'beta':-0.01*2.54, 'h':0.1, 'points':350,
         'n_Iapp':15, 'n_Iapp_new':15, 'I_max':0.1, 'I_min':0.0, 'resolution':1, 'n_sf':5,
         'fscale':'bicubic', 'bool_lineal':True, 'bool_scale':True,
         'generateIapp':False, 'generateNewIapp':False, 'shuffle':False}

def nagumo():
    param = dict(param0)#per evitar problemes de sobreescriptura pq param0 es una var global
    param['w0'] = -0.0320
    param['y0'] = 0.0812
    weights_y, weights_w = approximation(param, param['generateIapp'])
    param['w0'] = -0.03
    param['y0'] = 0.0
    if param['w_min']<=param['w0']<=param['w_max'] and param['y_min']<=param['y0']<=param['y_max']:
        ###SAVE THE NORMALIZED INPUT
        Wavenet.inputs(param['Iapp'])

        W_predict, Y_predict = simulation(param, weights_y, weights_w, param['generateNewIapp'])
        #display(param, W_predict, Y_predict)

        ###SAVE THE WEIGHTS IN A FILE
        #Wavenet.weights(weights_y, weights_w)
    else:
        print('ERROR: ICs for the simulation out of the trained range, try other ICs')

if __name__ == "__main__":
    from timeit import Timer
    print('Execution time: '+ str(Timer(lambda: nagumo()).timeit(number=1)) + ' s')
