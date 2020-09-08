from itertools import groupby
import numpy as np
import Wavenet
import random

### EULER
def euler(param, Iapp):
    ##Variables for saving the inputs
    y_train = np.zeros_like(Iapp)
    w_train = np.zeros_like(Iapp)
    #Variables for saving the targets
    target_y = np.zeros_like(Iapp)
    target_w = np.zeros_like(Iapp)
    #Parameters
    w0, y0 = param['w0'], param['y0'] #initial cond.
    h = param['h']
    a = param['a']
    alfa = param['alfa']
    beta = param['beta']
    n_points = param['points']
    
    Is = [key for key, group in groupby(Iapp)]
    i=0
    for I in Is:
        wn = np.zeros(n_points+1)
        yn = np.zeros(n_points+1)
        wn[0] = w0
        yn[0] = y0
        for k in range(n_points):
            y1 = y0 + h * (-y0*(y0-1)*(y0-a)+w0+I) #dv/dt = -f(v)+w-alpha*y+Ibase
            w1 = w0 + h * (alfa*y0+beta*w0) #dw/dt = eps*(-v-gamma*w)
            wn[k+1] = w1
            yn[k+1] = y1
            y0=y1
            w0=w1
        for j in range(n_points):
            target_w[i+j] = wn[1:][j]
            target_y[i+j] = yn[1:][j]
            w_train[i+j] = wn[:-1][j]
            y_train[i+j] = yn[:-1][j]
        i+=n_points
    return (target_w, w_train), (target_y, y_train)

### EXECUTE WAVENET
param0 = {'a':0.14, 'alfa':-0.01, 'beta':-0.01*2.54, 'h':0.1, 'points':1500,
         'n_Iapp':1500, 'n_Iapp_new':15, 'I_max':0.1, 'I_min':0.08, 'resolution':4, 'n_sf':5,
         'fscale':'bicubic', 'bool_lineal':True, 'bool_scale':True,
         'generateIapp':True, 'generateNewIapp':True, 'shuffle':False}

outputs = ['W_predict.txt', 'Y_predict.txt']
pesos = ['weights_w.txt', 'weights_y.txt']
        
###OBTAIN WEIGHTS
def approximation(param, all_data, index):
    redimensioned_target, input_data = Wavenet.posprocessed_data(param, all_data)
    Fx = Wavenet.matriu_Fx(param, input_data)
    weights = np.array([weight[0] for weight in Wavenet.training(param, Fx, redimensioned_target)])
    Wavenet.save_data(pesos[index], weights)
    
###EXECUTE THE SIMULATION FROM A FILE WITH INPUTS
def simulation(param, all_data, index):
    _, input_data = Wavenet.posprocessed_data(param, all_data)
    weights = Wavenet.read_weights(pesos[index])
    predicted_data = Wavenet.prediction(param, weights, input_data)
    Wavenet.save_data(outputs[index], predicted_data)

def Nagumo_2D():
    param = dict(param0)
    
    ## Memòria que es farà servir:
    if param['bool_scale']: neurones=1
    else: neurones=0
    for m in range(param['resolution']+1):
        neurones+=(2**m)
    print('Memoria necesaria:', round(param['points']*param['n_Iapp']*(1+param['n_sf']*neurones)*2.021518113*1e-5/1024, 2), 'GB aprox.')

    ## Aquí aproximo la funció
    if param['generateIapp']: Iapps = Wavenet.generate_data(param, param['n_Iapp'])
    else: Iapps = Wavenet.import_data(param)
    param['Iapp'] = Iapps

    ## Creating the grid of points with the all_data variable
    param['w0'] = -0.1
    param['y0'] = 0.0
    all_data = euler(param, Iapps)
    [approximation(param, all_data[i], i) for i in range(len(all_data))]

    ## Aquí predic el comportament
    if param['generateNewIapp']: Iapps = Wavenet.generate_data(param, param['n_Iapp_new'])
    else:
        Iapps = [Iapp for Iapp, group in groupby(param['Iapp'])]
        if param['shuffle']: random.shuffle(Iapps)
        Iapps = [Iapp for Iapp in Iapps for n_times in range(param['points'])]
    param['Iapp'] = Iapps

    param['w0'] = -0.1
    param['y0'] = 0.0
    all_data = euler(param, Iapps)
    [simulation(param, all_data[i], i) for i in range(len(all_data))]

    Wavenet.save_data('inputs.txt', param['Iapp'])
    ## Gràfics
    w, y = all_data
    visual(param, w[0], y[0], param['Iapp'])

def visual(param, w_target, y_target, Iapps):
    f1 = open('W_predict.txt', 'r')
    f2 = open('Y_predict.txt', 'r')
    w_predict = [float(line.replace('[', '').replace(']', '')) for line in f1]
    y_predict = [float(line.replace('[', '').replace(']', '')) for line in f2]
    f1.close()
    f2.close()
    phase_portrait(w_target, y_target, Iapps, w_predict, y_predict)
    answer = input("Show graphic? (y/n): ")
    if answer == 'y':
        time_graphic(w_target, param['Iapp'], w_predict, 'w output')
        time_graphic(y_target, param['Iapp'], y_predict, 'y output')

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

def phase_portrait(wout, yout, Iapps, w_predict, y_predict):
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title('FitzHugh-Nagumo 2D')
    ax.plot(w_predict, Iapps, y_predict, label='WNN', color='orange', marker='.', linestyle='--')
    ax.plot(wout, Iapps, yout, label='Target', color='blue', marker=',', linestyle='--')
    ax.set_xlabel('w')
    ax.set_ylabel('Iapp')
    ax.set_zlabel('y')
    ax.legend()    
    plt.show()

def time_graphic(target, Iapp, predicted_data, nom_ordenades):
    
    time=[]
    [time.append(i) for i in range(len(Iapp))]

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
