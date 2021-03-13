from configuration import config
from numba import njit, types
from numba.typed import Dict
import numpy as np
import Wavenet

############################################## EULER ##############################################

@njit(cache=True, fastmath=True, nogil=True)
def MLecar(dic, Iapps, w0=0.1, v0=-20):
    n_points = int(dic['points'])
    ##Variables for saving the inputs
    w_train = np.zeros(len(Iapps)*n_points)
    v_train = np.zeros(len(Iapps)*n_points)
    #Variables for saving the targets
    target_w = np.zeros(len(Iapps)*n_points)
    target_v = np.zeros(len(Iapps)*n_points)
    #Parameters
    h = dic['h']
    Cm = dic['Cm']
    phi = dic['phi']
    EL = dic['EL']
    EK = dic['EK']
    ECa = dic['ECa']
    V1 = dic['V1']
    V2 = dic['V2']
    V3 = dic['V3']
    V4 = dic['V4']
    gL = dic['gL']
    gK = dic['gK']
    gCa = dic['gCa']

    np.random.seed(1)
    noise = np.random.uniform(0, dic['noise'], len(Iapps)*n_points)    
    gK = gK*(1+noise)
    gL = gL*(1+noise)
    
    i=0
    for n,I in enumerate(Iapps):
        wn = np.zeros(n_points+1)
        vn = np.zeros(n_points+1)
        wn[0] = w0
        vn[0] = v0
        for k in range(n_points):
            # Funciones(v)
            minf = (1+np.tanh((v0-V1)/V2))/2
            winf = (1+np.tanh((v0-V3)/V4))/2
            tauinf = 1/(np.cosh((v0-V3)/(2*V4)))
            # Leakage current
            iL = gL[n_points*n:n_points*(n+1)][k]*(v0-EL)
            # Calcium current
            iCa =gCa*minf*(v0-ECa)
            # Potassium current
            iK = gK[n_points*n:n_points*(n+1)][k]*w0*(v0-EK)
            # ODE's system
            v1 = v0 + h * (1/Cm)*(-iL-iK-iCa+I)
            w1 = w0 + h * phi*(minf-w0)/tauinf
            wn[k+1] = w1
            vn[k+1] = v1
            v0=v1
            w0=w1
        for j in range(n_points):
            target_w[i+j] = wn[1:][j]
            target_v[i+j] = vn[1:][j]
            w_train[i+j] = wn[:-1][j]
            v_train[i+j] = vn[:-1][j]
        i+=n_points
    return (w_train, v_train), (target_w, target_v)

############################################## MAIN ##############################################

def WN(euler):
    ## Config
    param, euler_dict = config()
    var = ['w', 'v']
    titulo_graficos = 'Morris-Lecar 2D outputs'
    
    ## Find limits to later normalize
    Wavenet.domain_limits(param, Wavenet.dic(euler_dict), euler)
    ## Definir las cond_iniciales de la approx y la simu en cada funcion (pueden ser CI distintas)
    if not param['only_simulation']:
        Wavenet.approximation(param, euler_dict, euler, var, w0=0.17, v0=-30)
    Wavenet.prediction(param, euler_dict, euler, var, titulo_graficos, w0=0.17, v0=-30)

if __name__ == "__main__":
    from timeit import Timer
    print('\n', 'Execution time:', Timer(lambda: WN(MLecar)).timeit(number=1), 's')
