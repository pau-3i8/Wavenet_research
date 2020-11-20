from hyperlearn.numba import _sum, mean
from itertools import product, groupby
from tqdm import tqdm
import numpy as np
import Wavenet
import random
import psutil

### EULER
def euler(param, Iapp):
    ##Variables for saving the inputs
    v_train = np.zeros_like(Iapp)
    w_train = np.zeros_like(Iapp)
    y_train = np.zeros_like(Iapp)
    #Variables for saving the targets
    target_v = np.zeros_like(Iapp)
    target_w = np.zeros_like(Iapp)
    target_y = np.zeros_like(Iapp)
    #Parameters
    w0, v0, y0 = param['w0'], param['v0'], param['y0'] #initial cond.
    h = param['h']
    a = param['a']
    alpha = param['alpha']
    eps = param['eps']
    beta = param['beta']
    c = param['c']
    d = param['d']
    mu = param['mu']
    n_points = param['points']
    
    Is = [key for key, group in groupby(Iapp)]
    i=0
    for I in Is:
        wn = np.zeros(n_points+1)
        vn = np.zeros(n_points+1)
        yn = np.zeros(n_points+1)
        wn[0] = w0
        vn[0] = v0
        yn[0] = y0
        for k in range(n_points):
            v1 = v0 + h * (-v0*(v0-1)*(v0-a)+w0-alpha*y0+I) #dv/dt = -f(v)+w-alpha*y+Ibase
            w1 = w0 + h * (eps*v0+beta*w0) #dw/dt = eps*(-v-gamma*w)
            y1 = y0 + h * mu*(c-v0-d*y0) #dy/dt = mu*(c-v-d*y)
            wn[k+1] = w1
            vn[k+1] = v1
            yn[k+1] = y1
            v0=v1
            w0=w1
            y0=y1
        for j in range(n_points):
            target_w[i+j] = wn[1:][j]
            target_v[i+j] = vn[1:][j]
            target_y[i+j] = yn[1:][j]
            w_train[i+j] = wn[:-1][j]
            v_train[i+j] = vn[:-1][j]
            y_train[i+j] = yn[:-1][j]
        i+=n_points
    return (w_train, v_train, y_train), (target_w, target_v, target_y)

### EXECUTE WAVENET
param0 = {'a':0.14, 'eps':-0.01, 'beta':-0.01*2.54, 'alpha':0.05, 'c':-0.775, 'd':1,
          'mu':0.01, 'h':0.1, 'points':100, 'resolution':0, 'n_sf':5,
          'n_Iapp':10, 'n_Iapp_new':1, 'I_max':0.1, 'I_min':0.0,
          'fscale':'bicubic', 'bool_lineal':True, 'bool_scale':True,
          'generateIapp':True, 'generateNewIapp':False, 'shuffle':False}

outputs = ['W_predict.txt', 'V_predict.txt', 'Y_predict.txt']
pesos = ['weights_w.txt', 'weights_v.txt', 'weights_y.txt']
var = ['w', 'v', 'y']

### CREATE A POSTPROCESSED GRID OF POINTS WITH THE IAPPS
def posprocessed_data(param, train_data, Iapps):
    ## Save the domain limits to normalize the data within this limits
    param['max_1'] = 0.1#np.max(train_data[0])
    param['min_1'] = -0.4#np.min(train_data[0])
    param['max_2'] = 1.2#np.max(train_data[1])
    param['min_2'] = -0.4#np.min(train_data[1])
    param['max_3'] = -0.4#np.max(train_data[2])
    param['min_3'] = -1.8#np.min(train_data[2])
    ## Normalitze inputs
    input_1, input_2, input_3, Iapps = Wavenet.normalize(param, train_data[0], train_data[1], train_data[2], Iapps)
    return input_1, input_2, input_3, Iapps

###OBTAIN WEIGHTS
def approximation(param, all_data, Iapps):
    train_data, target_data = all_data
    input_1, input_2, input_3, Iapps = posprocessed_data(param, train_data, Iapps)
    Fx = Wavenet.matriu_Fx(param, input_1, input_2, input_3, Iapps)
    for i in range(len(target_data)):
        ## Adjust the dimension of the target vector for the WN
        target = np.array([[elem] for elem in list(target_data[i])])
        weights = np.array([weight[0] for weight in Wavenet.training(param, Fx, target, var[i])])
        Wavenet.save_data(pesos[i], weights)
    
###EXECUTE THE SIMULATION FROM A FILE WITH INPUTS
def simulation(param, target, Iapps, w0, v0, y0):
    weights_1 = Wavenet.read_weights(pesos[0])
    weights_2 = Wavenet.read_weights(pesos[1])
    weights_3 = Wavenet.read_weights(pesos[2])
    target_1, target_2, target_3 = target
    input_1, input_2, input_3 = [np.array([w0]), np.array([v0]), np.array([y0])]
    predicted_1 = np.zeros_like(target[0])
    predicted_2 = np.zeros_like(target[1])
    predicted_3 = np.zeros_like(target[2])
    for j,I in enumerate(tqdm(Iapps, desc='Predicció completada', unit='integracions', leave=False)):
        #reaprofito el max i el min de les dades entrenades
        input_1, input_2, input_3, I = Wavenet.normalize(param, input_1, input_2, input_3, np.array([I]))
        Fx = Wavenet.matriu_Fx(param, input_1, input_2, input_3, I)
        ##Fi Fx
        input_1 = (Fx.dot(np.array(weights_1)))[0] #wn+1
        input_2 = (Fx.dot(np.array(weights_2)))[0] #vn+1
        input_3 = (Fx.dot(np.array(weights_3)))[0] #yn+1
        predicted_1[j] = input_1
        predicted_2[j] = input_2
        predicted_3[j] = input_3
    print('RMSE', var[0], ':', _sum((target_1-predicted_1)**2)/_sum((mean(target_1)-target_1)**2)*100,'%')
    print('RMSE', var[1], ':', _sum((target_2-predicted_2)**2)/_sum((mean(target_2)-target_2)**2)*100,'%')
    print('RMSE', var[2], ':', _sum((target_3-predicted_3)**2)/_sum((mean(target_3)-target_3)**2)*100,'%')
    Wavenet.save_data(outputs[0], predicted_1)
    Wavenet.save_data(outputs[1], predicted_2)
    Wavenet.save_data(outputs[2], predicted_3)
    
def Nagumo_2D():
    param = dict(param0)
    ## Memòria que es farà servir:
    if param['bool_scale']: neurones=1
    else: neurones=0
    for m in range(param['resolution']+1):
        neurones+=(2**(4*m)+4*2**(3*m)+6*2**(2*m)+4*2**m)
    memoria = round(param['points']*param['n_Iapp']*(3+(param['n_sf']**4)*neurones)*2.021518113*1e-5/1024, 2)
    print('Memoria necesaria:', memoria, 'GB aprox.')
    if memoria <= psutil.virtual_memory().total/(1024**3):
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
    param['y0'] = -1.
    all_data = euler(param, param['Iapp'])
    Wavenet.save_data('inputs.txt', param['Iapp'])
    approximation(param, all_data, param['Iapp'])
    
    ## Aquí predic el comportament (simulo)
    if param['generateNewIapp']: Iapps = Wavenet.generate_data(param, param['n_Iapp_new'])
    else:
        Iapps = [Iapp for Iapp, group in groupby(param['Iapp'])][:param['n_Iapp_new']]
        if param['shuffle']: random.shuffle(Iapps)
        Iapps = [Iapp for Iapp in Iapps for n_times in range(param['points'])]
    param['Iapp'] = Iapps
    param['w0'] = -0.1
    param['v0'] = 0.0
    param['y0'] = -1.
    _, target = euler(param, param['Iapp'])
    simulation(param, target, param['Iapp'], param['w0'], param['v0'], param['y0'])
    
    ## Gràfics
    w, v, y = target
    visual(param, w, v, y, param['Iapp'])

def visual(param, w_target, v_target, y_target, Iapps):
    f1 = open('W_predict.txt', 'r')
    f2 = open('V_predict.txt', 'r')
    f3 = open('Y_predict.txt', 'r')
    w_predict = [float(line.replace('[', '').replace(']', '')) for line in f1]
    v_predict = [float(line.replace('[', '').replace(']', '')) for line in f2]
    y_predict = [float(line.replace('[', '').replace(']', '')) for line in f3]
    f1.close()
    f2.close()
    f3.close()
    phase_portrait(w_target, v_target, y_target, Iapps, w_predict, v_predict, y_predict)
    #answer = input("Show graphic? (y/n): ")
    #if answer == 'y':
    time_graphic(w_target, param['Iapp'], w_predict, 'w output')
    time_graphic(v_target, param['Iapp'], v_predict, 'v output')
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

def phase_portrait(wout, vout, yout, Iapps, w_predict, v_predict, y_predict):
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title('FitzHugh-Nagumo 3D')
    ax.plot(wout, yout, vout, label='Target', color='blue', marker='.', linestyle='')
    ax.plot(w_predict, y_predict, v_predict, label='WNN', color='orange', marker=',', linestyle='')
    ax.set_xlabel('w')
    ax.set_ylabel('y')
    ax.set_zlabel('v')
    ax.legend()    
    plt.show()

def time_graphic(target, Iapp, predicted_data, nom_ordenades):
    
    time=[]
    [time.append(i) for i in range(len(target))]
    
    plt.figure()
    plt.subplot(211)
    plt.title('FitzHugh-Nagumo 3D')
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
