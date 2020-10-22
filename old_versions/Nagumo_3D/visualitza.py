import Wavenet_3D as Wavenet
import Nagumo_3D as Nagumo
from numba import njit

def read_weights():
    f1 = open('weights_y.txt', 'r')
    f2 = open('weights_w.txt', 'r')
    f3 = open('weights_z.txt', 'r')
    weights_y = [[float(line.replace('[', '').replace(']', ''))] for line in f1]
    weights_w = [[float(line.replace('[', '').replace(']', ''))] for line in f2]
    weights_z = [[float(line.replace('[', '').replace(']', ''))] for line in f3]
    f1.close()
    f2.close()
    f3.close()
    return weights_y, weights_w, weights_z

def save_outputs(W_predict, Y_predict, Z_predict): #m'estalvio de refer F(x)
    f1 = open('W_predict.txt', 'w')
    f2 = open('Y_predict.txt', 'w')
    f3 = open('Z_predict.txt', 'w')
    [f1.writelines([str(W_predict[i][0])+'\n']) for i in range(len(W_predict))]
    [f2.writelines([str(Y_predict[i][0])+'\n']) for i in range(len(Y_predict))]
    [f3.writelines([str(Z_predict[i][0])+'\n']) for i in range(len(Z_predict))]
    f1.close()
    f2.close()
    f3.close()

param1 = {'a':0.14, 'alfa':-0.01, 'beta':-0.01*2.54, 'eta':0.0, 'c':-0.775, 'd':1, 'kappa':0.1,
        'h':0.1, 'points':2000}

def visual():
    param = dict(param1)
    f1 = open('W_predict.txt', 'r')
    f2 = open('Y_predict.txt', 'r')
    f3 = open('Z_predict.txt', 'r')
    W_predict = [[float(line.replace('[', '').replace(']', ''))] for line in f1]
    Y_predict = [[float(line.replace('[', '').replace(']', ''))] for line in f2]
    Z_predict = [[float(line.replace('[', '').replace(']', ''))] for line in f3]
    f1.close()
    f2.close()
    f3.close()
    param['w0'] = -0.02
    param['y0'] = -0.01
    param['z0'] = -0.77
    Iapps = Nagumo.import_data(param)
    target_y, target_w, target_z, _, _, _, Iapp = training_data(param, Iapps)
    Wavenet.graphic(target_w, target_y, target_z, W_predict, Y_predict, Z_predict)
    Wavenet.graphic_time(target_w, target_y, target_z, Iapp, W_predict, Y_predict, Z_predict)


#per saber si la matriu es dispersa:
import matplotlib.pyplot as plt
def dispersa():
    param = {'a':0.14, 'alfa':-0.01, 'beta':-0.01*2.54, 'eta':0.05, 'c':-0.775, 'd':1, 'kappa':0.01,
             'h':0.1, 'points':5000, 'n_Iapp':2, 'n_Iapp_new':2, 'I_max':0.1, 'I_min':0.0,
             'resolution':0, 'n_sf':5, 'fscale':'bicubic', 'bool_lineal':True, 'bool_scale':True}

    param['w0'] = -0.1
    param['y0'] = 0.0
    param['z0'] = -0.8
    print('Construint matriu')
    Iapps = Nagumo.generate_data(param, param['n_Iapp'])
    _, _, _, y_train, w_train, z_train, Iapp = Nagumo.training_data(param, Iapps)
    _, wavelet = Nagumo.matriu_Fx(param, y_train, w_train, z_train, Iapp)

    #wavelet=matriu
    print('Contant zeros')
    contador(wavelet)
    fig = plot_coo_matrix(wavelet)
    fig.savefig('matriu_dispersa.png')

from scipy.sparse import coo_matrix

def plot_coo_matrix(m):
    if not isinstance(m, coo_matrix):
        m = coo_matrix(m)
    fig = plt.figure()
    ax = fig.add_subplot(111, facecolor='white')
    ax.plot(m.col, m.row, ',', color='black', ms=0.1) #black=zeros
    ax.set_xlim(0, m.shape[1])
    ax.set_ylim(0, m.shape[0])
    #ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.invert_yaxis()
    print('guardant imatge')
    return fig

@njit(fastmath = True, nogil = True, cache = True)
def contador(wavelet):
    cont0 = 0
    files, cols = wavelet.shape #cal que sigui un array
    mida = files*cols
    
    for i in range(files):
        for j in range(cols):
            valor = abs(wavelet[i][j])
            if valor == 0: cont0 += 1
            
    if cont0 > (mida/2):
        print('Matriu dispersa', 'mida =', mida, 'Dispersa =', cont0/mida*100)
    else:
        print('No es dispersa')
        
dispersa()
