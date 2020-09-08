from grafic_normes import grafic_normes
from grafic_pesos import grafic_pesos
from grafic_relatiu import grafic_relatiu
from matplotlib import pylab
from itertools import product
import sensibilitat as sen
import numpy as np

pylab.rcParams['figure.figsize'] = 20,20
pylab.rcParams['axes.titlesize'] = 20
pylab.rcParams['axes.labelsize'] = 16
pylab.rcParams['xtick.labelsize'] = 14
pylab.rcParams['ytick.labelsize'] = 14
pylab.rcParams['legend.fontsize'] = 14
pylab.rcParams['axes.labelpad'] = 5
pylab.rcParams['axes.titlepad'] = 10

### aquest diccionari 'param' puc treure-li algunes keys perquè no em calen, per trobar els pesos
param = {'a':0.14, 'alfa':-0.01, 'gamma':2.54, 'h':0.1, 'points':1500,
         'n_Iapp':1800, 'I_max':0.25, 'I_min':0.0, 'resolution':1, 'n_sf':5,
         'fscale':'bicubic', 'bool_lineal':True, 'bool_scale':True,
         'generateIapp':True, 'generateNewIapp':False, 'shuffle':False}

#configuració de totes les possibles combinacions
num_as = 5
num_gammas = 5

###Execusió
def exe(param, num_as, num_gammas):
    param['w0'] = -0.0320 #condicions de contorn
    param['y0'] = 0.0812
    #estudi de la 'a' i la 'gamma' per separat
    a = np.linspace(0.1, 0.2, num_as)
    a = np.append(a, [(0.14)], axis=0)
    gamma = np.linspace(2, 3, num_gammas)
    gamma = np.append(gamma, [(2.54)], axis=0)
    configs = list(product(a,[(2.54)]))+list(product([(0.14)], gamma))[:-1]
    sen.log_Is_limit(param, configs)
    l = sen.sensibilitat(param, configs)
    sen.log_weights(l)

def grafics(param):
    grafic_normes(param)
    pylab.rcParams['axes.labelpad'] = 20
    pylab.rcParams['axes.titlepad'] = 15
    grafic_pesos(param)
    grafic_relatiu(param)

#exe(param, num_as, num_gammas)
grafics(param)
