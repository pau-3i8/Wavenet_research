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
    for I in Is:
        wn = np.zeros(n_points+1)
        vn = np.zeros(n_points+1)
        wn[0] = w0
        vn[0] = v0
        for k in range(n_points):
            v1 = v0 + h * (-v0*(v0-1)*(v0-a)+w0+I) #dv/dt = -f(v)+w+Ibase
            w1 = w0 + h * (eps*v0+beta*w0) #dw/dt = eps*(-v-gamma*w)
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

### EXECUTE WAVENET
param0 = {'a':0.14, 'eps':-0.01, 'beta':-0.01*2.54, 'h':0.1, 'points':150,
         'n_Iapp':100, 'n_Iapp_new':1, 'I_max':0.1, 'I_min':0.0, 'resolution':0, 'n_sf':5,
         'fscale':'bicubic', 'bool_lineal':True, 'bool_scale':True,
         'generateIapp':True, 'generateNewIapp':False, 'shuffle':False}

outputs = ['W_predict.txt', 'V_predict.txt']
pesos = ['weights_w.txt', 'weights_v.txt']
var = ['w', 'v']
        
###OBTAIN WEIGHTS
def approximation(param, all_data, Iapps):
    train_data, target_data = all_data
    input_1, input_2, Iapps = Wavenet.posprocessed_data(param, train_data, Iapps)
    Fx = Wavenet.matriu_Fx(param, input_1, input_2 ,Iapps)
    for i in range(len(target_data)):
        ## Adjust the dimension of the target vector for the WN
        target = np.array([[elem] for elem in list(target_data[i])])
        weights = np.array([weight[0] for weight in Wavenet.training(param, Fx, target, var[i])])
        Wavenet.save_data(pesos[i], weights)
    
###EXECUTE THE SIMULATION FROM A FILE WITH INPUTS
def simulation(param, target, Iapps, w0, v0):
    weights_1 = Wavenet.read_weights(pesos[0])
    weights_2 = Wavenet.read_weights(pesos[1])
    target_1, target_2 = target
    input_1, input_2 = [np.array([w0]), np.array([v0])]
    predicted_1 = np.zeros_like(target[0])
    predicted_2 = np.zeros_like(target[1])
    for j,I in enumerate(tqdm(Iapps, desc='Predicció completada', unit='integracions', leave=False)):
        #reaprofito el max i el min de les dades entrenades
        input_1, input_2, I = Wavenet.normalize(param, input_1, input_2, np.array([I], dtype='float32'))
        Fx = Wavenet.matriu_Fx(param, input_1, input_2, I)
        ##Fi Fx
        input_1 = (Fx.dot(np.array(weights_1)))[0] #wn+1
        input_2 = (Fx.dot(np.array(weights_2)))[0] #yn+1
        predicted_1[j] = input_1
        predicted_2[j] = input_2
    print('RMSE', var[0], ':', _sum((target_1-predicted_1)**2)/_sum((mean(target_1)-target_1)**2)*100,'%')
    print('RMSE', var[1], ':', _sum((target_2-predicted_2)**2)/_sum((mean(target_2)-target_2)**2)*100,'%')
    Wavenet.save_data(outputs[0], predicted_1)
    Wavenet.save_data(outputs[1], predicted_2)
    
def Nagumo_2D():
    param = dict(param0)
    ## Memòria que es farà servir:
    if param['bool_scale']: neurones=1
    else: neurones=0
    for m in range(param['resolution']+1):
        neurones+=(2**(3*m)+3*2**(2*m)+3*2**m)
    memoria = round(param['points']*param['n_Iapp']*(2+(param['n_sf']**3)*neurones)*2.021518113*1e-5/1024, 2)
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
    _, target = euler(param, param['Iapp'])
    simulation(param, target, param['Iapp'], param['w0'], param['v0'])
    
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
    #answer = input("Show graphic? (y/n): ")
    #if answer == 'y':
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
    ax.plot(w_predict, Iapps, v_predict, label='WNN', color='orange', marker='.', linestyle='--')
    ax.plot(wout, Iapps, vout, label='Target', color='blue', marker=',', linestyle='--')
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
    plt.title('FitzHugh-Nagumo 2D')
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
