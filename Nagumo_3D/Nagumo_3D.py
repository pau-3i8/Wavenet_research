from hyperlearn.random import uniform
from hyperlearn.numba import _sum, mean
from itertools import product, groupby
import Wavenet_3D as Wavenet
import numpy as np
import random

###CREATE IAPPS
def generate_data(param, n_Iapps):
    #Iapp = np.random.uniform(param['I_min'], param['I_max'], n_Iapps)
    Iapp = uniform(param['I_min'], param['I_max'], n_Iapps)
    Iapp = [key for key, group in groupby(Iapp) for n_times in range(param['points'])]
    return Iapp

###IMPORT IAPPS FROM A FILE
def import_data(param):
    Iapp = Wavenet.read_inputs('inputs.txt')
    Iapp = [key for key, group in groupby(Iapp) for n_times in range(param['points'])]
    return Iapp

###CREATE GRID OF POINTS WITH THE IAPPS
def training_data(param, Iapps):

    w0, y0, z0 = param['w0'], param['y0'],  param['z0']#initial cond.
    Iapp = np.array(Iapps)
    ##Variables for saving the inputs
    y_train = np.zeros_like(Iapp)
    w_train = np.zeros_like(Iapp)
    z_train = np.zeros_like(Iapp)
    ##Variables for saving the targets
    target_y = np.zeros_like(Iapp)
    target_w = np.zeros_like(Iapp)
    target_z = np.zeros_like(Iapp)
    ##Obtaining the target
    
    h = param['h']
    a = param['a']
    eta = param['eta']
    alfa = param['alfa']
    beta = param['beta']
    c = param['c']
    d = param['d']
    kappa = param['kappa']
    n_points = param['points']
    
    Is = [key for key, group in groupby(Iapp)]
    target_w, target_y, target_z, w_train, y_train, z_train = Wavenet.euler(np.array(Is), w0, y0, z0, n_points, h, a, eta, alfa, beta, c, d, kappa, target_w, target_y, target_z, w_train, y_train, z_train)
    
    ##Adjust the dimension of the target vector for the WN
    tary = np.array([[elem] for elem in list(target_y)])
    tarw = np.array([[elem] for elem in list(target_w)])
    tarz = np.array([[elem] for elem in list(target_z)])
    
    ##Save the domain limits to normalize the data within this limits
    param['I_max'] = np.max(Iapp)
    param['I_min'] = np.min(Iapp)
    param['y_max'] = np.max(y_train)
    param['y_min'] = np.min(y_train)
    param['w_max'] = np.max(w_train)
    param['w_min'] = np.min(w_train)
    param['z_max'] = np.max(z_train)
    param['z_min'] = np.min(z_train)
    ##Save the non normalized variables for graphics
    param['Iapp'] = Iapp
    param['target_y'] = target_y
    param['target_w'] = target_w
    param['target_z'] = target_z
    
    ##Normalitze inputs
    y_train, w_train, z_train, Iapp = Wavenet.normalize(param, y_train, w_train, z_train, Iapp)
    
    return tary, tarw, tarz, y_train, w_train, z_train, Iapp

###TRAINING
def matriu_Fx(param, y_train, w_train, z_train, Iapp):
    
    ###LINEAL COMP. MATRIX
    wavelet = Wavenet.lineal(w_train, y_train, z_train)
    
    mult = param['n_sf']**4 #j**E 4 inputs
    Nh = 0
    if param['bool_lineal']:
        Nh += len(wavelet[0])
    if param['bool_scale']:
        Nh += mult
        ###SCALE FUNC. MATRIX
        wavelet = Wavenet.scale(param['fscale'], w_train, y_train, z_train, Iapp, param['n_sf'], wavelet, param['bool_lineal'])
    
    for m in range(param['resolution']+1):
        Nh += mult*(4*(2**(m))+6*(2**(2*m))+4*(2**(3*m))+1*(2**(4*m))) #wavelon's growth
        ###WAVELET'S MATRIX 
        wavelet = Wavenet.wavelets(param['fscale'], w_train, y_train, z_train, Iapp, m, wavelet, param['n_sf'])
        
    return Nh, wavelet

def training(param, Nh, wavelet, target):
    ###LOSS
    weights = Wavenet.loss(wavelet, target, Nh)
    ###COMPUTE ERROR
    print('MSE at level = ' + str(param['resolution']+2) + ' is: ' + str(_sum((target - np.dot(wavelet, weights.T))**2)/len(target)))
    ###RETURN WEIGHTS
    return weights.T

###OBTAIN WEIGHTS
def approximation(param, generate):
    if generate:
        Iapps = generate_data(param, param['n_Iapp'])
        #generes un numero de Iapps n_Iapp per aproximar la funcio
    if not generate:
        Iapps = import_data(param)
    tary, tarw, tarz, y_train, w_train, z_train, Iapp = training_data(param, Iapps)
    
    ##Variables for storing the calculated weights
    print('Building F(x)')
    Nh, wavelet = matriu_Fx(param, y_train, w_train, z_train, Iapp)
    print('Computing weights')

    weights_y = np.array([weight for weight in training(param, Nh, wavelet, tary)])
    weights_w = np.array([weight for weight in training(param, Nh, wavelet, tarw)])
    weights_z = np.array([weight for weight in training(param, Nh, wavelet, tarz)])
    
    return weights_y, weights_w, weights_z

###PREDICTION
def prediction(param, weights_y, weights_w, weights_z, y_train, w_train, z_train, Iapp):
    print('Building new F(x) for prediction')
    wavelet = Wavenet.lineal(w_train, y_train, z_train)
    if param['bool_scale']:
        wavelet = Wavenet.scale(param['fscale'], w_train, y_train, z_train, Iapp, param['n_sf'], wavelet, param['bool_lineal'])
    
    for m in range(param['resolution']+1):
        wavelet = Wavenet.wavelets(param['fscale'], w_train, y_train, z_train, Iapp, m, wavelet, param['n_sf'])

    Y_predict = (np.dot(wavelet, weights_y)).T[0]
    W_predict = (np.dot(wavelet, weights_w)).T[0]
    Z_predict = (np.dot(wavelet, weights_z)).T[0]
    
    ###PREDICTION'S ERROR
    y = param['target_y']
    print('Prediction RMSE: ' + str((np.sum((y-Y_predict)**2)/np.sum((np.ones((1,len(y)))*mean(y)-y)**2))*100) + '%')
    
    return W_predict, Y_predict, Z_predict

###DISPLAY THE RESULT
def display(param, W_predict, Y_predict, Z_predict):
    answer = input("Show graphic? (y/n): ")
    if answer == 'y':
        Wavenet.graphic(param['target_w'], param['target_y'], param['target_z'], W_predict, Y_predict, Z_predict)
        Wavenet.graphic_time(param['target_w'], param['target_y'], param['target_z'], param['Iapp'], W_predict, Y_predict, Z_predict)
    
###EXECUTE THE SIMULATION FROM A FILE WITH INPUTS
def simulation(param, weights_y, weights_w, weights_z, generate_new):
    if generate_new:
        Iapps = generate_data(param, param['n_Iapp_new'])
    if not generate_new:
        Iapps = [Iapp for Iapp, group in groupby(param['Iapp'])]
        if param['shuffle']:
            random.shuffle(Iapps)
        Iapps = [Iapp for Iapp in Iapps for n_times in range(param['points'])]
        
    _, _, _, y_train, w_train, z_train, Iapp = training_data(param, Iapps)
        
    W_predict, Y_predict, Z_predict = prediction(param, weights_y, weights_w, weights_z, y_train, w_train, z_train, Iapp)
    return W_predict, Y_predict, Z_predict

### EXECUTE WAVENET
param0 = {'a':0.14, 'alfa':-0.01, 'beta':-0.01*2.54, 'eta':0.05, 'c':-0.775, 'd':1, 'kappa':0.01,
         'h':0.1, 'points':5000, 'n_Iapp':15, 'n_Iapp_new':10, 'I_max':0.1, 'I_min':0.09,
         'resolution':0, 'n_sf':5, 'fscale':'bicubic', 'bool_lineal':True, 'bool_scale':True,
         'generateIapp':False, 'generateNewIapp':False, 'shuffle':False}

def nagumo():
    param = dict(param0)
    param['w0'] = -0.1
    param['y0'] = 0.0
    param['z0'] = -0.8
    weights_y, weights_w, weights_z = approximation(param, param['generateIapp'])
    param['w0'] = -0.15
    param['y0'] = 0.04
    param['z0'] = -0.8
    if param['w_min']<=param['w0']<=param['w_max'] and param['y_min']<=param['y0']<=param['y_max'] and param['z_min']<=param['z0']<=param['z_max']:
        ###SAVE THE NORMALIZED INPUT
        Wavenet.inputs(param['Iapp'])

        W_predict, Y_predict, Z_predict = simulation(param, weights_y, weights_w, weights_z, param['generateNewIapp'])
        display(param, W_predict, Y_predict, Z_predict)

        ###SAVE THE WEIGHTS IN A FILE
        Wavenet.weights(weights_y, weights_w, weights_z)
        ###SAVE THE OUTPUTS IN A FILE
        #visualitza.save_outputs(W_predict, Y_predict, Z_predict)
    else:
        print('ERROR: ICs for the simulation out of the trained range, try other ICs')

if __name__ == "__main__":
    from timeit import Timer
    print('Execution time: '+ str(Timer(lambda: nagumo()).timeit(number=1)) + ' s')
