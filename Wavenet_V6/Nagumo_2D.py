from configuration import config
from numba import njit, types
from numba.typed import Dict
import numpy as np
import Wavenet

############################################## EULER ##############################################

@njit(cache=True, fastmath=True, nogil=True)
def Nagumo_2D(dic, Iapps, w0=0, v0=0):
    n_points = int(dic['points'])
    ##Variables for saving the inputs
    w_train = np.zeros(len(Iapps)*n_points)
    v_train = np.zeros(len(Iapps)*n_points)
    #Variables for saving the targets
    target_w = np.zeros(len(Iapps)*n_points)
    target_v = np.zeros(len(Iapps)*n_points)
    #Parameters
    h = dic['h']
    a = dic['a']
    eps = dic['eps']
    gamma = dic['gamma']
    
    i=0
    for I in Iapps:
        wn = np.zeros(n_points+1)
        vn = np.zeros(n_points+1)
        wn[0] = w0
        vn[0] = v0
        for k in range(n_points):
            v1 = v0 + h * (-v0*(v0-1)*(v0-a)+w0+I) #dv/dt = -f(v)+w+Ibase
            w1 = w0 + h * eps*(-v0-gamma*w0)       #dw/dt = eps*(-v-gamma*w)
            wn[k+1] = w1                           #f(v) = v*(v-1)*(v-a)
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

############################################## MAIN ##############################################

def WN(euler):
    ## Config
    param, euler_dict = config()
    var = ['w', 'y']
    titulo_graficos = 'FitzHugh-Nagumo 2D outputs'
    
    ## Find limits to later normalize
    Wavenet.domain_limits(param, Wavenet.dic(euler_dict), euler)
    ## Definir las cond_iniciales de la approx y la simu en cada funcion (pueden ser CI distintas)
    if not param['only_simulation']:
        Wavenet.approximation(param, euler_dict, euler, var, w0=-0.1, v0=0.0)
    Wavenet.prediction(param, euler_dict, euler, var, titulo_graficos, w0=-0.11, v0=0.11)

if __name__ == "__main__":
    from timeit import Timer
    print('\n', 'Execution time:', Timer(lambda: WN(Nagumo_2D)).timeit(number=1), 's')
