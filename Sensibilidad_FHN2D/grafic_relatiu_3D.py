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

def relatiu(nou, ref):            
    return np.sqrt((ref-nou)**2)

def grafic_relatiu():
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

    ya_irel = []
    wa_irel = []
    
    yref = relatiu(np.array(ya_ref), np.array(ya_ref))
    ya_first = relatiu(np.array(ya_first), np.array(ya_ref))
    ya_last = relatiu(np.array(ya_last), np.array(ya_ref))
    for elem in ya_i:
        ya_irel.append(relatiu(np.array(elem), np.array(ya_ref)))

    wref = relatiu(np.array(wa_ref), np.array(wa_ref))
    wa_first = relatiu(np.array(wa_first), np.array(wa_ref))
    wa_last = relatiu(np.array(wa_last), np.array(wa_ref))
    for elem in wa_i:
        wa_irel.append(relatiu(np.array(elem), np.array(wa_ref)))

    yg_irel = []
    wg_irel = []

    yg_first = relatiu(np.array(yg_first), np.array(ya_ref))
    yg_last = relatiu(np.array(yg_last), np.array(ya_ref))
    for elem in yg_i:
        yg_irel.append(relatiu(np.array(elem), np.array(ya_ref)))

    wg_first = relatiu(np.array(wg_first), np.array(wa_ref))
    wg_last = relatiu(np.array(wg_last), np.array(wa_ref))
    for elem in wg_i:
        wg_irel.append(relatiu(np.array(elem), np.array(ya_ref)))
    
    x = np.arange(1, len(ya_first)+1, 1)
        
    fig = plt.figure(1)
    ax = plt.axes(projection='3d')
    plt.title('Sensibilidad y_a') #canviar el gruix de les linies amb lw=num
    ax.plot3D(x, 5*np.ones(len(x)), yref, color='green', label='referencia')
    ax.plot3D(x, 0*np.ones(len(x)), ya_first, color='orange', label='a=0.1 gamma=2.54')
    for i,elem in enumerate(ya_irel):
        ax.plot3D(x, (i+1)*np.ones(len(x)), elem, color='red', label='config ' + str(i+1))
    ax.plot3D(x, 4*np.ones(len(x)), ya_last, color='blue', label='a=0.2 gamma=2.54')
    ax.set_xlabel('Dimensión de la base')
    ax.set_ylabel('Congiguraciones')
    ax.set_zlabel('Valor relativo pesos yn+1')
    ax.legend(loc='center left', bbox_to_anchor=(-.1,0.5))
    
    fig = plt.figure(2)
    ax = plt.axes(projection='3d')
    plt.title('Sensibilidad w_a') #canviar el gruix de les linies amb lw=num
    ax.plot3D(x, 5*np.ones(len(x)), wref, color='green', label='referencia')
    ax.plot3D(x, 0*np.ones(len(x)), wa_first, color='orange', label='a=0.1 gamma=2.54')
    for i,elem in enumerate(wa_irel):
        ax.plot3D(x, (i+1)*np.ones(len(x)), elem, color='red', label='config ' + str(i+1))
    ax.plot3D(x, 4*np.ones(len(x)), wa_last, color='blue', label='a=0.2 gamma=2.54')
    ax.set_xlabel('Dimensión de la base')
    ax.set_ylabel('Congiguraciones')
    ax.set_zlabel('Valor relativo pesos wn+1')
    ax.legend(loc='center left', bbox_to_anchor=(-.1,0.5))
    
    fig = plt.figure(3)
    ax = plt.axes(projection='3d')
    plt.title('Sensibilidad y_gamma') #canviar el gruix de les linies amb lw=num
    ax.plot3D(x, 5*np.ones(len(x)), yref, color='green', label='referencia')
    ax.plot3D(x, 0*np.ones(len(x)), yg_first, color='orange', label='a=0.14 gamma=2.0')
    for i,elem in enumerate(yg_irel):
        ax.plot3D(x, (i+1)*np.ones(len(x)), elem, color='red', label='config ' + str(i+1))
    ax.plot3D(x, 4*np.ones(len(x)), yg_last, color='blue', label='a=0.14 gamma=3.0')
    ax.set_xlabel('Dimensión de la base')
    ax.set_ylabel('Congiguraciones')
    ax.set_zlabel('Valor relativo pesos yn+1')
    ax.legend(loc='center left', bbox_to_anchor=(-.1,0.5))
    
    fig = plt.figure(4)
    ax = plt.axes(projection='3d')
    plt.title('Sensibilidad w_gamma') #canviar el gruix de les linies amb lw=num
    ax.plot3D(x, 5*np.ones(len(x)), wref, color='green', label='referencia')
    ax.plot3D(x, 0*np.ones(len(x)), wg_first, color='orange', label='a=0.14 gamma=2.0')
    for i,elem in enumerate(wg_irel):
        ax.plot3D(x, (i+1)*np.ones(len(x)), elem, color='red', label='config ' + str(i+1))
    ax.plot3D(x, 4*np.ones(len(x)), wg_last, color='blue', label='a=0.14 gamma=3.0')
    ax.set_xlabel('Dimensión de la base')
    ax.set_ylabel('Congiguraciones')
    ax.set_zlabel('Valor relativo pesos wn+1')    
    ax.legend(loc='center left', bbox_to_anchor=(-.1,0.5))
    
    plt.show()


grafic_relatiu()
