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
    h_train = np.zeros_like(Iapp)
    n_train = np.zeros_like(Iapp)
    #Variables for saving the targets
    target_v = np.zeros_like(Iapp)
    target_h = np.zeros_like(Iapp)
    target_n = np.zeros_like(Iapp)
    #Parameters
    v0, h0, n0 = param['v0'], param['h0'], param['n0'] #initial cond.
    gK = param['gK']
    gNa = param['gNa']
    gL = param['gL']
    vK = param['vK']
    vNa = param['vNa']
    vL = param['vL']
    phi = param['phi']
    h = param['h']
    n_points = param['points']
    
    Is = [key for key, group in groupby(Iapp)]
    i=0
    for I in Is:
        vn = np.zeros(n_points+1)
        hn = np.zeros(n_points+1)
        nn = np.zeros(n_points+1)
        vn[0] = v0
        hn[0] = h0
        nn[0] = n0
        for k in range(n_points):
            #leak current
            iL = gL*(v0-vL)
            #sodium current
            am = -0.1*(v0+33.0)/(np.exp(-0.1*(v0+33.0))-1.0)
            bm = 4.0*np.exp(-(v0+58.0)/12.0)
            minf = am/(am+bm)
            iNa = gNa*(minf**3)*h0*(v0-vNa)
            ah = 0.07*np.exp(-(v0+50.0)/10.0)
            bh = 1/(1+np.exp(-0.1*(v0+20.0)))
            #delayed rectifier - potassium current
            iK = gK*(n0**4)*(v0-vK)
            an = -0.01*(v0+34.0)/(np.exp(-0.1*(v0+34.0))-1.0)
            bn = 0.125*np.exp(-(v0+44.0)/25.0)
            #Ionic currents
            iIon = iNa+iK
            #Sistema d'eq's
            v1 = v0 + h * (-iL-iIon+I)
            h1 = h0 + h * (phi*(ah*(1-h0)-bh*h0))
            n1 = n0 + h * (phi*(an*(1-n0)-bn*n0))
            vn[k+1] = v1
            hn[k+1] = h1
            nn[k+1] = n1
            v0=v1
            h0=h1
            n0=n1
        for j in range(n_points):
            target_v[i+j] = vn[1:][j]
            target_h[i+j] = hn[1:][j]
            target_n[i+j] = nn[1:][j]
            v_train[i+j] = vn[:-1][j]
            h_train[i+j] = hn[:-1][j]
            n_train[i+j] = nn[:-1][j]
        i+=n_points    
    return (v_train, h_train, n_train), (target_v, target_h, target_n)

### EXECUTE WAVENET
#Rang d'estudi Iapps: -3 a 3
#Fer servir un h=0.1 peta tot, faig servir exponencials massa grans perquè no està refinada l'aprox.
param0 = {'gK':18.0, 'gNa':45.0, 'gL':0.1, 'vK':-80.0, 'vNa':55.0, 'vL':-65.0, 'phi':4.0,
          'h':0.005, 'points':100, 'resolution':0, 'n_sf':5, 'n_Iapp':10, 'n_Iapp_new':1,
          'I_max':3.0, 'I_min':-3.0, 'fscale':'bicubic', 'bool_lineal':True, 'bool_scale':True,
          'generateIapp':True, 'generateNewIapp':False, 'shuffle':False}

outputs = ['V_predict.txt', 'H_predict.txt', 'N_predict.txt']
pesos = ['weights_v.txt', 'weights_h.txt', 'weights_n.txt']
var = ['v', 'h', 'n']

### CREATE A POSTPROCESSED GRID OF POINTS WITH THE IAPPS
def posprocessed_data(param, train_data, Iapps):
    ## Save the domain limits to normalize the data within this limits
    #Per evitar sustos és més fàcils fixar els límits coneixent el model.
    param['max_1'] = 60.#np.max(train_data[0])
    param['min_1'] = -100.#np.min(train_data[0])
    param['max_2'] = 1.2#np.max(train_data[1])
    param['min_2'] = 0.#np.min(train_data[1])
    param['max_3'] = 0.9#np.max(train_data[2])
    param['min_3'] = -0.2#np.min(train_data[2])
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
def simulation(param, target, Iapps, v0, h0, n0):
    weights_1 = Wavenet.read_weights(pesos[0])
    weights_2 = Wavenet.read_weights(pesos[1])
    weights_3 = Wavenet.read_weights(pesos[2])
    target_1, target_2, target_3 = target
    input_1, input_2, input_3 = [np.array([v0]), np.array([h0]), np.array([n0])]
    predicted_1 = np.zeros_like(target[0])
    predicted_2 = np.zeros_like(target[1])
    predicted_3 = np.zeros_like(target[2])
    for j,I in enumerate(tqdm(Iapps, desc='Predicció completada', unit='integracions', leave=False)):
        #reaprofito el max i el min de les dades entrenades
        input_1, input_2, input_3, I = Wavenet.normalize(param, input_1, input_2, input_3, np.array([I]))
        Fx = Wavenet.matriu_Fx(param, input_1, input_2, input_3, I)
        ##Fi Fx
        input_1 = (Fx.dot(np.array(weights_1)))[0]
        input_2 = (Fx.dot(np.array(weights_2)))[0]
        input_3 = (Fx.dot(np.array(weights_3)))[0]
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
    param['v0'] = -55.0
    param['h0'] = 0.83
    param['n0'] = 0.11
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
    param['v0'] = -55.0
    param['h0'] = 0.83
    param['n0'] = 0.11
    _, target = euler(param, param['Iapp'])
    simulation(param, target, param['Iapp'], param['v0'], param['h0'], param['n0'])
    
    ## Gràfics
    v, h, n = target
    visual(param, v, h, n, param['Iapp'])

def visual(param, v_target, h_target, n_target, Iapps):
    f1 = open('V_predict.txt', 'r')
    f2 = open('H_predict.txt', 'r')
    f3 = open('N_predict.txt', 'r')
    v_predict = [float(line.replace('[', '').replace(']', '')) for line in f1]
    h_predict = [float(line.replace('[', '').replace(']', '')) for line in f2]
    n_predict = [float(line.replace('[', '').replace(']', '')) for line in f3]
    f1.close()
    f2.close()
    f3.close()
    phase_portrait(v_target, h_target, n_target, Iapps, v_predict, h_predict, n_predict)
    #answer = input("Show graphic? (y/n): ")
    #if answer == 'y':
    time_graphic(v_target, param['Iapp'], v_predict, 'v output')
    time_graphic(h_target, param['Iapp'], h_predict, 'h output')
    time_graphic(n_target, param['Iapp'], n_predict, 'n output')

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

def phase_portrait(vout, hout, nout, Iapps, v_predict, h_predict, n_predict):
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title('FitzHugh-Nagumo 3D')
    ax.plot(vout, hout, nout, label='Target', color='blue', marker='.', linestyle='')
    ax.plot(v_predict, h_predict, n_predict, label='WNN', color='orange', marker=',', linestyle='')
    ax.set_xlabel('v')
    ax.set_ylabel('h')
    ax.set_zlabel('n')
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
