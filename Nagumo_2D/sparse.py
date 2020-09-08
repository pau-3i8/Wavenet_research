import matplotlib.pyplot as plt
import Wavenet as Wavenet
import Nagumo as Nagumo
from numba import njit

#per saber si la matriu es dispersa:
def dispersa():
    param = {'a':0.14, 'alfa':-0.01, 'beta':-0.01*2.54, 'h':0.1, 'points':1500,
             'n_Iapp':15, 'I_max':0.1, 'I_min':0.0, 'resolution':1, 'n_sf':5,
             'fscale':'bicubic', 'bool_lineal':True, 'bool_scale':True,}
    
    param['w0'] = -0.03
    param['y0'] = 0.0
    print('Construint matriu')
    Iapps = Nagumo.generate_data(param, param['n_Iapp'])
    _, _, y_train, w_train, Iapp = Nagumo.training_data(param, Iapps)
    _, wavelet = Nagumo.matriu_Fx(param, y_train, w_train, Iapp)

    #wavelet=matriu
    print('Contant zeros')
    contador(wavelet)
    #guardo en imatges perquè visualitzar-ho directament es menja la RAM
    fig = plot_coo_matrix(wavelet)
    fig.savefig('matriu_dispersa.png')
    fig = plot_coo_matrix((wavelet.T).dot(wavelet))
    fig.savefig('covariancia_dispersa.png')
    
from scipy.sparse import coo_matrix

def plot_coo_matrix(m):
    if not isinstance(m, coo_matrix):
        m = coo_matrix(m)
    fig = plt.figure()
    ax = fig.add_subplot(111, facecolor='white')
    ax.plot(m.col, m.row, ',', color='black', ms=0.001) #black = zeros
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
    files, cols = wavelet.shape #cal que sigui un array
    mida = files*cols
    
    cont0 = 0
    for i in range(files):
        for j in range(cols):
            valor = abs(wavelet[i][j])
            if valor == 0: cont0 += 1

    if cont0 > (mida/2):
        print('Matriu dispersa', 'mida =', mida, 'Dispersa =', cont0/mida*100)
    else:
        print('No es dispersa')
        
dispersa()
