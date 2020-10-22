from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
from numpy.linalg import eigvals
from scipy.optimize import fsolve
import numpy as np
import sympy as sp

#configura vigualització, format dels gràfics.
pylab.rcParams['figure.figsize'] = 14, 12
pylab.rcParams['axes.titlesize'] = 20
pylab.rcParams['axes.labelsize'] = 16
pylab.rcParams['xtick.labelsize'] = 14
pylab.rcParams['ytick.labelsize'] = 14
pylab.rcParams['legend.fontsize'] = 14
sns.set_style('ticks')
palette = ["#1E88E5", "#43A047", "#e53935", "#5E35B1", "#FFB300", "#00ACC1", "#3949AB", "#F4511E"]
sns.set_palette(palette)

param0 = {'a':0.14, 'alfa':-0.01, 'beta':-0.01*2.54, 'Iapp':None,
         'eta':0.0, 'c':-0.775, 'd':1, 'kappa':0.1, 'h':0.1}
#sistema solucionable: eta=0.05
# si kappa=0.01, n_points=5.000 || si kappa=0.001, n_points=50.000
    
def yprima(y, w, z, u, param): #u=Iapp
    a = param['a']
    eta = param['eta']
    return -y*(y-1)*(y-a) + w - eta*z + u #dv/dt = -f(v) + w - alpha* y + Ibase
                                          #f(v) = v*(v-1)*(v-a)

def wprima(y, w, param):
    alfa = param['alfa']
    beta = param['beta']
    return alfa*y+beta*w #dw/dt = eps*(-v-gamma*w)

def zprima(y, z, param):
    c = param['c']  # Al mail es -0.775
    d = param['d']  # Segons les dades és 1, una mica trivial
    kappa = param['kappa'] # Al mail es 0.001
    return kappa*(c - y - d*z) #dy/dt = mu*(c-v-d*y)

def euler(u, w0, y0, z0, param, n_points):
    h = param['h']
    wn = [w0]
    yn = [y0]
    zn = [z0]
    for i in range(n_points):
        y1 = y0 + h * yprima(y0, w0, z0, u, param)
        w1 = w0 + h * wprima(y0, w0, param) 
        z1 = z0 + h * zprima(y0, z0, param)
        wn.append(w1)
        yn.append(y1)
        zn.append(z1)
        y0 = y1
        w0 = w1
        z0 = z1
    return wn[1:], yn[1:], zn[1:]#, wn[:-1], yn[:-1], zn[:-1]

def edos(cond_ini, param):
    w, y, z = cond_ini[0], cond_ini[1], cond_ini[2]
    #constants
    a = param['a']
    eta = param['eta']
    alfa = param['alfa']
    beta = param['beta']
    c = param['c']
    d = param['d']
    kappa = param['kappa']
    u = param['Iapp'] #la Iapp
    #EDOS
    dw = alfa*y+beta*w
    dy = -y*(y-1)*(y-a) + w - eta*z + u
    dz = kappa*(c - y - d*z)
    
    return [dw, dy, dz]

def calc_vaps(init, param):
    #init són les cond inicials que es generen per usarse a cond_ini
    
    #Symbolic eq
    (w, y, z) = sp.symbols('w, y, z', real = True)
    dw, dy, dz= edos((w, y, z), param)
    #Jacobià
    J = sp.Matrix([dw, dy, dz]).jacobian([w, y, z])
    #Punts estables
    w_ini, y_ini, z_ini = fsolve(edos, init, args=(param,))
    p_estables = [w_ini, y_ini, z_ini]
    J_aPE = J.subs([(w, w_ini), (y, y_ini), (z, z_ini)])
    J_aPE = np.array(J_aPE).astype(np.float64)
    vaps = eigvals(J_aPE)
    return vaps, p_estables

def graf(param, n_points):
    param = dict(param0)
    # resolució per vaps per diferents Iapps to demonstrate bifurcation behaviors
    init = [-0.1, -0, -0.775] #primeres con.inicials [w,y,z]
    w0, y0, z0 = [-0.15, 0.4, -0.8]
    u = np.arange(0.03, 0.05, 0.01) # 10 us
    #u = np.random.uniform(0, 0.3, 30)
    
    w = []
    y = []
    z = []

    ww=[]
    yy=[]
    zz=[]
    uu=[]

    for Iapp in u:
        param['Iapp'] = Iapp
        vaps, p_estables = calc_vaps(init, param)
        w.append(p_estables[0])
        y.append(p_estables[1])
        z.append(p_estables[2])
        #al ser punts estables, aplicar euler a aquest punts, a cada h la oslucio es la mateixa
        #perque les edos convergeixen al punt estable
        wn1, yn1, zn1= euler(Iapp, w0, y0, z0, param, n_points)
        ww.append(wn1)
        yy.append(yn1)
        zz.append(zn1)
        uu.append(Iapp)
        w0=wn1[-1]
        y0=yn1[-1]
        z0=zn1[-1]

    ww = [e for elem in ww for e in elem]
    yy = [e for elem in yy for e in elem]
    zz = [e for elem in zz for e in elem]
    uu = [elem for n_points in range(n_points) for elem in uu]
    
    ##Tot i ser 3 edos representaré w, y, u en un gràfic 3D
    fig = plt.figure(1)
    ax = plt.axes(projection='3d')
    ax.scatter(w,  y, z) #PE
    ax.plot(np.array(ww), np.array(yy), np.array(zz), marker=',', linestyle='-') #Punts Euler
    ax.set_xlabel('w')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    
graf(param0, 5000) #freq Iapp = n_points
