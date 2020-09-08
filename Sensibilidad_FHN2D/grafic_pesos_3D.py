from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import pylab
import numpy as np

pylab.rcParams['figure.figsize'] = 20,20
pylab.rcParams['axes.titlesize'] = 20
pylab.rcParams['axes.labelsize'] = 16
pylab.rcParams['xtick.labelsize'] = 14
pylab.rcParams['ytick.labelsize'] = 14
pylab.rcParams['legend.fontsize'] = 14
pylab.rcParams['axes.labelpad'] = 20
pylab.rcParams['axes.titlepad'] = 40

def grafic_pesos():
    data=[]
    p = []
    f = open('sensibilitat.txt', 'r')
    for linia in f:
        linia = linia.split()
        for elem in linia[1:-2]:
            p.append(float(elem))
        linia = [linia[0], p, linia[-2], linia[-1]]
        p=[]
        data.append(linia)
    f.close()

    ya_i = []
    wa_i = []
    yg_i = []
    wg_i = []
    for elem in data:
        if elem[-2] == '0.14' and elem[-1] == '2.54' and elem[0] == 'y':
            ya_ref = elem[1]
            yg_ref = ya_ref
        elif elem[-2] == '0.1' and elem[-1] == '2.54' and elem[0] == 'y':
            ya_first = elem[1]
        elif elem[-2] == '0.2' and elem[-1] == '2.54' and elem[0] == 'y':
            ya_last = elem[1]
        elif elem[-2] != ('0.1' and '0.2') and elem[-1] == '2.54' and elem[0]=='y':
            ya_i.append(elem[1])
        elif elem[-2] == '0.14' and elem[-1] == '2.0' and elem[0] == 'y':
            yg_first = elem[1]
        elif elem[-2] == '0.14' and elem[-1] == '3.0' and elem[0] == 'y':
            yg_last = elem[1]
        elif elem[-1] != ('2.0' and '3.0') and elem[-2] == '0.14' and elem[0]=='y':
            yg_i.append(elem[1])

        if elem[-2] == '0.14' and elem[-1] == '2.54' and elem[0] == 'w':
            wa_ref = elem[1]
            wg_ref = wa_ref
        elif elem[-2] == '0.1' and elem[-1] == '2.54' and elem[0] == 'w':
            wa_first = elem[1]
        elif elem[-2] == '0.2' and elem[-1] == '2.54' and elem[0] == 'w':
            wa_last = elem[1]
        elif elem[-2] != ('0.1' and '0.2') and elem[-1] == '2.54' and elem[0]=='w':
            wa_i.append(elem[1])
        elif elem[-2] == '0.14' and elem[-1] == '2.0' and elem[0] == 'w':
            wg_first = elem[1]
        elif elem[-2] == '0.14' and elem[-1] == '3.0' and elem[0] == 'w':
            wg_last = elem[1]
        elif elem[-1] != ('2.0' and '3.0') and elem[-2] == '0.14' and elem[0]=='w':
            wg_i.append(elem[1])
    
    x = np.arange(1, len(ya_first)+1, 1)
    
    fig = plt.figure(1)
    ax = plt.axes(projection='3d')
    plt.title('Sensibilidad y_a') #canviar el gruix de les linies amb lw=num
    ax.plot3D(x, 5*np.ones(len(x)), ya_ref, color='green', label='referencia')
    ax.plot3D(x, 0*np.ones(len(x)), ya_first, color='orange', label='a=0.1 gamma=2.54')
    for i,elem in enumerate(ya_i):
        ax.plot3D(x, (i+1)*np.ones(len(x)), elem, color='red', label='config ' + str(i+1))
    ax.plot3D(x, 4*np.ones(len(x)), ya_last, color='blue', label='a=0.2 gamma=2.54')
    ax.set_xlabel('Dimensión de la base')
    ax.set_ylabel('Congiguraciones')
    ax.set_zlabel('Pesos yn+1')
    ax.legend(loc='center left', bbox_to_anchor=(-.1,0.5))
    
    fig = plt.figure(2)
    ax = plt.axes(projection='3d')
    plt.title('Sensibilidad w_a') #canviar el gruix de les linies amb lw=num
    ax.plot3D(x, 5*np.ones(len(x)), wa_ref, color='green', label='referencia')
    ax.plot3D(x, 0*np.ones(len(x)), wa_first, color='orange', label='a=0.1 gamma=2.54')
    for i,elem in enumerate(wa_i):
        ax.plot3D(x, (i+1)*np.ones(len(x)), elem, color='red', label='config ' + str(i+1))
    ax.plot3D(x, 4*np.ones(len(x)), wa_last, color='blue', label='a=0.2 gamma=2.54')
    ax.set_xlabel('Dimensión de la base')
    ax.set_ylabel('Congiguraciones')
    ax.set_zlabel('Pesos wn+1')
    ax.legend(loc='center left', bbox_to_anchor=(-.1,0.5))
    
    fig = plt.figure(3)
    ax = plt.axes(projection='3d')
    plt.title('Sensibilidad y_gamma') #canviar el gruix de les linies amb lw=num
    ax.plot3D(x, 5*np.ones(len(x)), yg_ref, color='green', label='referencia')
    ax.plot3D(x, 0*np.ones(len(x)), yg_first, color='orange', label='a=0.14 gamma=2.0')
    for i,elem in enumerate(yg_i):
        ax.plot3D(x, (i+1)*np.ones(len(x)), elem, color='red', label='config ' + str(i+1))
    ax.plot3D(x, 4*np.ones(len(x)), yg_last, color='blue', label='a=0.14 gamma=3.0')
    ax.set_xlabel('Dimensión de la base')
    ax.set_ylabel('Congiguraciones')
    ax.set_zlabel('Pesos yn+1')
    ax.legend(loc='center left', bbox_to_anchor=(-.1,0.5))
    
    fig = plt.figure(4)
    ax = plt.axes(projection='3d')
    plt.title('Sensibilidad w_gamma') #canviar el gruix de les linies amb lw=num
    ax.plot3D(x, 5*np.ones(len(x)), wg_ref, color='green', label='referencia')
    ax.plot3D(x, 0*np.ones(len(x)), wg_first, color='orange', label='a=0.14 gamma=2.0')
    for i,elem in enumerate(wg_i):
        ax.plot3D(x, (i+1)*np.ones(len(x)), elem, color='red', label='config ' + str(i+1))
    ax.plot3D(x, 4*np.ones(len(x)), wg_last, color='blue', label='a=0.14 gamma=3.0')
    ax.set_xlabel('Dimensión de la base')
    ax.set_ylabel('Congiguraciones')
    ax.set_zlabel('Pesos wn+1')    
    ax.legend(loc='center left', bbox_to_anchor=(-.1,0.5))
    
    plt.show()

    #ax = fig.add_subplot(1, 1, 1, projection='3d')
    """
    X = np.outer(np.arange(1, len(ya_first)+1, 1), np.ones(len(ya_i))).T
    Y = (np.ones((len(x),len(ya_i)))*np.arange(1, len(ya_i)+1, 1)).T
    Z = np.array(ya_i)
    """
    #ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
    #ax.contourf(X, Y, Z)
    #ax.plot_surface(X, Y, Z)
    """
    X1=[e for elem in X for e in elem]
    Y1=[e for elem in Y for e in elem]
    Z1=[e for elem in Z for e in elem]
    """    
    #ax.plot_trisurf(X1, Y1, Z1)
    """
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.3)
    cset = ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
    cset = ax.contour(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
    cset = ax.contour(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)
    """
    
grafic_pesos()
