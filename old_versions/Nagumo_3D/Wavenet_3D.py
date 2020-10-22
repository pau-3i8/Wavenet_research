from scale_func import select_phi_scaled, select_psi
from itertools import product
from numba import njit
import numpy as np

def scale_fun(nom, x, n):
    return select_phi_scaled(nom, x, n)
def mother_wavelet(nom, x, n):
    return select_psi(nom, x, n)

def lineal(y_train, w_train, z_train):
    linealy = (np.ones((1, len(y_train)))*y_train).T
    linealw = (np.ones((1, len(y_train)))*w_train).T
    linealz = (np.ones((1, len(z_train)))*z_train).T
    lineal = np.append(linealy, linealw, axis=1)
    lineal = np.append(lineal, linealz, axis=1)
    return lineal
    
def scale(nom, y_train, w_train, z_train, Iapp, n_sf, lineal, bool_lineal):
    wavelets = np.zeros((len(y_train), n_sf**4)).T
    n = [n for n in range(n_sf)]
    for i,elem in enumerate(list(product(n,n,n,n))):
        n1, n2, n3, n4 = elem
        wavelets[i] = scale_fun(nom, y_train, n1)*scale_fun(nom, w_train, n2)*scale_fun(nom, z_train, n3)*scale_fun(nom, Iapp, n4)

    wavelets = wavelets.T
    if bool_lineal:
        wavelets = np.append(lineal, wavelets, axis=1)
    return wavelets

### Version_3
def wavelets(nom, y_train, w_train, z_train, Iapp, m, wavelet, n_sf):
    N = len(y_train)
    n = [n for n in range(n_sf)]
    c = [c for c in range(2**m)]
    v1 = [(y_train, w_train, z_train, Iapp), (Iapp, y_train, w_train, z_train), (z_train, Iapp, y_train, w_train), (w_train, z_train, Iapp, y_train)]
    v2 = [(y_train, w_train, z_train, Iapp), (Iapp, y_train, w_train, z_train), (z_train, Iapp, y_train, w_train), (w_train, z_train, Iapp, y_train), (z_train, y_train, w_train, Iapp), (w_train, Iapp, z_train, y_train)]
    
    aux = np.zeros((N, (n_sf**4)*(2**(4*m)+4*2**(3*m)+6*2**(2*m)+4*2**m))).T
    i = 0
    for ns in list(product(n,n,n,n)):
        n1, n2, n3, n4 = ns
        for var in v1:
            for c1 in c:
                aux[i] = scale_fun(nom, var[0], n1)* scale_fun(nom, var[1], n2)* scale_fun(nom, var[2], n3)* mother_wavelet(nom, (2**m)* var[3] - c1, n4)
                i+=1
        for var in v2:
            for ci in list(product(c,c)):                
                c1, c2 = ci
                aux[i] = scale_fun(nom, var[0], n1)* scale_fun(nom, var[1], n2)* mother_wavelet(nom, (2**m)* var[2] - c1, n3)* mother_wavelet(nom, (2**m)* var[3] - c2, n4)
                i+=1
        for var in v1:
            for ci in list(product(c,c,c)):
                c1, c2, c3 = ci
                aux[i] = mother_wavelet(nom, (2**m)* var[0] - c1, n1)* mother_wavelet(nom, (2**m)* var[1] - c2, n2)* mother_wavelet(nom, (2**m)* var[2] - c3, n3)* scale_fun(nom, var[3], n4)
                i+=1
        for ci in list(product(c,c,c,c)):
            c1, c2, c3, c4 = ci
            aux[i] = mother_wavelet(nom, (2**m)* y_train - c1, n1)* mother_wavelet(nom, (2**m)* w_train - c2, n2)* mother_wavelet(nom, (2**m)* z_train - c3, n3)* mother_wavelet(nom, (2**m)* Iapp - c4, n4)
            i+=1
            
    wavelet = np.append(wavelet, aux.T, axis=1)
    return wavelet

############################################ FUNCTIONS ############################################

### EULER
@njit(nogil=True, fastmath=True)
def euler(Is, w0, y0, z0, n_points, h, a, eta, alfa, beta, c, d, kappa, target_w, target_y, target_z, w_train, y_train, z_train):
    i=0
    for I in Is:
        wn = np.zeros(n_points+1)
        yn = np.zeros(n_points+1)
        zn = np.zeros(n_points+1)
        wn[0] = w0
        yn[0] = y0
        zn[0] = z0
        for k in range(n_points):
            y1 = y0 + h * (-y0*(y0-1)*(y0-a)+w0-eta*z0+I) #dv/dt = -f(v)+w-alpha*y+Ibase
            w1 = w0 + h * (alfa*y0+beta*w0) #dw/dt = eps*(-v-gamma*w)
            z1 = z0 + h * kappa*(c-y0-d*z0) #dy/dt = mu*(c-v-d*y)
            wn[k+1] = w1
            yn[k+1] = y1
            zn[k+1] = z1
            y0=y1
            w0=w1
            z0=z1
        for j in range(n_points):
            target_w[i+j] = wn[1:][j]
            target_y[i+j] = yn[1:][j]
            target_z[i+j] = zn[1:][j]
            w_train[i+j] = wn[:-1][j]
            y_train[i+j] = yn[:-1][j]
            z_train[i+j] = zn[:-1][j]
        i+=n_points
    return target_w, target_y, target_z, w_train, y_train, z_train

### LOSS
from hyperlearn.solvers import lstsq, solve

def loss(wavelet, target, rank):
    """
    Solve és més ràpid que lstsq per matrius molt grans i tots dos tenen una precisió semblant,
    però per matrius més petites, tot i que solve sigui més rapid lstsq és més precís.
    Caldria fer un anàlisi complet per veure què interessa més en el nostre cas, però tenint en
    compte que el temps prima i que treballarem sempre amb matriu amb milions de files, la funció
    solve de hyperlearn és la millor opció.
    """
    ###Fastest | less accurate
    return np.array([solve(wavelet,target, alpha=1e-6)])
    ###Second fastest | most accurate
    #return lstsq(wavelet, target).T

### NORMALITZATION
def normalize(param, y, w, z, I):
    ynorm = (y-param['y_min'])/(param['y_max']-param['y_min'])
    wnorm = (w-param['w_min'])/(param['w_max']-param['w_min'])
    znorm = (z-param['z_min'])/(param['z_max']-param['z_min'])
    Inorm = (I-param['I_min'])/(param['I_max']-param['I_min'])
    return ynorm, wnorm, znorm, Inorm
        
### DOCS
def inputs(Iapp):
    f = open('inputs.txt', 'w')
    [f.writelines([str(Iapp[i])+'\n']) for i in range(len(Iapp))]
    f.close()

def weights(sigma_y, sigma_w, sigma_z):
    f1 = open('weights_y.txt', 'w')
    f2 = open('weights_w.txt', 'w')
    f3 = open('weights_z.txt', 'w')
    [f1.writelines([str(sigma_y[i])+'\n']) for i in range(len(sigma_y))]
    [f2.writelines([str(sigma_w[i])+'\n']) for i in range(len(sigma_y))]
    [f3.writelines([str(sigma_z[i])+'\n']) for i in range(len(sigma_y))]
    f1.close()
    f2.close()
    f3.close()
    
def read_inputs(doc):
    f = open(doc, 'r')
    Iapp = [float(line.split()[0]) for line in f]
    f.close()
    return Iapp

### GRAPHIC
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

def graphic(wout, yout, zout, w_predict, y_predict, z_predict):
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title('FitzHugh-Nagumo 3D')
    ax.plot(wout, zout, yout, label='Target', color='blue', marker='.', linestyle='')
    ax.plot(w_predict, z_predict, y_predict, label='WNN', color='orange', marker=',', linestyle='')
    ax.set_xlabel('w')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.legend()    
    plt.show()

def graphic_time(wout, yout, zout, Iapp, w_predict, y_predict, z_predict):
    
    time=[]
    [time.append(i) for i in range(len(Iapp))]

    fig = plt.figure()
    plt.subplot(221)
    plt.title('FitzHugh-Nagumo 3D')
    plt.xlabel('Steps')
    plt.ylabel('y outputs')
    plt.plot(time, yout, label='Target', color='blue', marker=',', linestyle='-')
    plt.plot(time, y_predict, label='WNN', color='orange', marker=',', linestyle='-')
    plt.legend()

    plt.subplot(222)
    plt.title('FitzHugh-Nagumo 3D')
    plt.xlabel('Steps')
    plt.ylabel('w outputs')
    plt.plot(time, wout, label='Target', color='blue', marker=',', linestyle='-')
    plt.plot(time, w_predict, label='WNN', color='orange', marker=',', linestyle='-')
    plt.legend()

    plt.subplot(223)
    plt.title('FitzHugh-Nagumo 3D')
    plt.xlabel('Steps')
    plt.ylabel('z outputs')
    plt.plot(time, zout, label='Target', color='blue', marker=',', linestyle='-')
    plt.plot(time, z_predict, label='WNN', color='orange', marker=',', linestyle='-')
    plt.legend()
    
    plt.subplot(224)
    plt.title('Randomised forced term')
    plt.xlabel('Steps')
    plt.ylabel('Iapp')
    plt.plot(time, Iapp, color='blue', marker=',', linestyle='-')
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.10, right=0.95, hspace=0.5, wspace=0.35)
    plt.show()
