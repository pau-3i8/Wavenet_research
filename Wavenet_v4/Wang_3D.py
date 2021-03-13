from configuration import config
from numba import njit, types
from numba.typed import Dict
import numpy as np
import Wavenet

############################################## EULER ##############################################

@njit(cache=True, fastmath=True, nogil=True)
def Wang_3D(dic, Iapps, v0=-50, h0=0, n0=0):
    n_points = int(dic['points'])
    ##Variables for saving the inputs
    v_train = np.zeros(len(Iapps)*n_points)
    h_train = np.zeros(len(Iapps)*n_points)
    n_train = np.zeros(len(Iapps)*n_points)
    #Variables for saving the targets
    target_v = np.zeros(len(Iapps)*n_points)
    target_h = np.zeros(len(Iapps)*n_points)
    target_n = np.zeros(len(Iapps)*n_points)
    #Parameters
    h = dic['h']
    gK = dic['gK']
    gNa = dic['gNa']
    gL = dic['gL']
    vK = dic['vK']
    vNa = dic['vNa']
    vL = dic['vL']
    phi = dic['phi']
    
    i=0
    for I in Iapps:
        vn = np.zeros(n_points+1)
        hn = np.zeros(n_points+1)
        nn = np.zeros(n_points+1)
        vn[0] = v0
        hn[0] = h0
        nn[0] = n0
        for k in range(n_points):
            # leakage current (iL)
            iL = gL*(v0-vL)
            # sodium current (iNa)
            am = -0.1*(v0+33.0)/(np.exp(-0.1*(v0+33.0))-1.0)
            bm = 4.0*np.exp(-(v0+58.0)/12.0)
            minf = am/(am+bm)
            iNa = gNa*(minf**3)*h0*(v0-vNa)
            ah = 0.07*np.exp(-(v0+50.0)/10.0)
            bh = 1/(1+np.exp(-0.1*(v0+20.0)))
            # delayed rectifier - potassium current (iK)
            iK = gK*(n0**4)*(v0-vK)
            an = -0.01*(v0+34.0)/(np.exp(-0.1*(v0+34.0))-1.0)
            bn = 0.125*np.exp(-(v0+44.0)/25.0)
            # Ionic currents (iIon)
            iIon = iNa+iK
            # ODE's system
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

############################################## MAIN ##############################################

def WN(euler):
    ## Config
    param, euler_dict = config()
    var = ['v', 'h', 'n']
    titulo_graficos = 'Wang 3D outputs'
    
    ## Find limits to later normalize
    Wavenet.domain_limits(param, Wavenet.dic(euler_dict), euler)
    ## Definir las cond_iniciales de la approx y la simu en cada funcion (pueden ser CI distintas)
    if not param['only_simulation']:
        Wavenet.approximation(param, euler_dict, euler, var, v0=-55.0, h0=0.83, n0=0.11)
    Wavenet.prediction(param, euler_dict, euler, var, titulo_graficos, v0=-55.0, h0=0.83, n0=0.11)

if __name__ == "__main__":
    from timeit import Timer
    print('\n', 'Execution time:', Timer(lambda: WN(Wang_3D)).timeit(number=1), 's')
