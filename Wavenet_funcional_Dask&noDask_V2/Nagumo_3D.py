from configuration import config
from numba import njit, types
from numba.typed import Dict
import numpy as np
import Wavenet

############################################## EULER ##############################################

@njit(cache=True, fastmath=True, nogil=True)
def Nagumo_3D(dic, Iapps, w0=0, v0=0, y0=0):
    n_points = int(dic['points'])
    ##Variables for saving the inputs
    v_train = np.zeros(len(Iapps)*n_points)
    w_train = np.zeros(len(Iapps)*n_points)
    y_train = np.zeros(len(Iapps)*n_points)
    #Variables for saving the targets
    target_v = np.zeros(len(Iapps)*n_points)
    target_w = np.zeros(len(Iapps)*n_points)
    target_y = np.zeros(len(Iapps)*n_points)  
    #Parameters
    h = dic['h']
    a = dic['a']
    alpha = dic['alpha']
    eps = dic['eps']
    gamma = dic['gamma']
    c = dic['c']
    d = dic['d']
    mu = dic['mu']
    
    i=0
    for I in Iapps:
        wn = np.zeros(n_points+1)
        vn = np.zeros(n_points+1)
        yn = np.zeros(n_points+1)
        wn[0] = w0
        vn[0] = v0
        yn[0] = y0
        for k in range(n_points):
            v1 = v0 + h * (-v0*(v0-1)*(v0-a)+w0-alpha*y0+I) #dv/dt = -f(v)+w-alpha*y+Ibase
            w1 = w0 + h * eps*(-v0-gamma*w0)                #dw/dt = eps*(-v-gamma*w)
            y1 = y0 + h * mu*(c-v0-d*y0)                    #dy/dt = mu*(c-v-d*y)
            wn[k+1] = w1                                    #f(v) = v*(v-1)*(v-a)
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

############################################## MAIN ##############################################

def WN(euler):
    ## Config
    param, euler_dict = config()
    var = ['w', 'y', 'v']
    titulo_graficos = 'FitzHugh-Nagumo 3D outputs'
    
    ## Find limits to later normalize
    Wavenet.domain_limits(param, Wavenet.dic(euler_dict), euler)
    """com que la funció amb l'euler té valors default, no cal declarar cap i només és per obtenir màxims i mínims per després normalitzar les dades, però si es volgués canviar les condicions inicials caldria definir: domain_limits(param, Nagumo_3D, w0=algo, v0=algo, y0=algo) (no cal definir el 3 punts)"""
    ## Definir las cond_iniciales de la approx y la simu en cada funcion (pueden ser CI distintas)
    if not param['only_simulation']:
        Wavenet.approximation(param, euler_dict, euler, var, w0=-0.1, v0=0.0, y0=-1.0)
    Wavenet.prediction(param, euler_dict, euler, var, titulo_graficos, w0=-0.1, v0=0.0, y0=-1.0)

if __name__ == "__main__":
    from timeit import Timer
    print('\n', 'Execution time:', Timer(lambda: WN(Nagumo_3D)).timeit(number=1), 's')
