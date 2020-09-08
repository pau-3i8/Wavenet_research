import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pylab

pylab.rcParams['figure.figsize'] = 20,20
pylab.rcParams['axes.titlesize'] = 20
pylab.rcParams['axes.labelsize'] = 16
pylab.rcParams['xtick.labelsize'] = 14
pylab.rcParams['ytick.labelsize'] = 14
pylab.rcParams['legend.fontsize'] = 14
pylab.rcParams['axes.labelpad'] = 5
pylab.rcParams['axes.titlepad'] = 10

def norma(l):
    suma = 0
    for elem in l:
        suma+=(elem)**2
    return np.sqrt(suma)

def grafic_normes(param):
    
    lineal=[]
    nivell1=[]
    nivell2=[]
    nivell3=[]
    s = param['n_sf']**3
    
    #segregat
    f = open('sensibilitat.txt', 'r')
    for linia in f:
        linia = linia.split()
        pesos = [float(elem) for elem in linia[1:-2]]        
        lineal.append([linia[0], pesos[0:2], linia[-2], linia[-1]])
        nivell1.append([linia[0], pesos[2:s+2], linia[-2], linia[-1]])
        nivell2.append([linia[0], pesos[s+2:(7*s)+(s+2)], linia[-2], linia[-1]])
        nivell3.append([linia[0], pesos[(7*s)+(s+2):(s*26)+(7*s)+(s+2)], linia[-2], linia[-1]])
    f.close()

    lin_ya=[]
    lvl1_ya=[]
    lvl2_ya=[]
    lvl3_ya=[]
    lin_wa=[]
    lvl1_wa=[]
    lvl2_wa=[]
    lvl3_wa=[]
    lin_yg=[]
    lvl1_yg=[]
    lvl2_yg=[]
    lvl3_yg=[]
    lin_wg=[]
    lvl1_wg=[]
    lvl2_wg=[]
    lvl3_wg=[]
    
    for elem in lineal:
        ####Primer guardaré les simus amb els paràmetres de referència i després la resta
        #perquè no es barregin
        if elem[-2] == '0.14' and elem[-1] == '2.54' and elem[0] == 'y':
            lin_ya.append((norma(elem[1]), elem[-2]))
            lin_yg.append((norma(elem[1]), elem[-1]))
        elif elem[-2] == '0.14' and elem[-1] == '2.54' and elem[0] == 'w':
            lin_wa.append((norma(elem[1]), elem[-2]))
            lin_wg.append((norma(elem[1]), elem[-1]))
        #ara la resta (a cada loop igual)
        elif '0.1' <= elem[-2] <= '0.2' and elem[-1] == '2.54' and elem[0] == 'y':
            lin_ya.append((norma(elem[1]), elem[-2]))
        elif '0.1' <= elem[-2] <= '0.2' and elem[-1] == '2.54' and elem[0] == 'w':
            lin_wa.append((norma(elem[1]), elem[-2]))
        elif elem[-2] == '0.14' and '2.0' <= elem[-1] <= '3.0' and elem[0] == 'y':
            lin_yg.append((norma(elem[1]), elem[-1]))
        elif elem[-2] == '0.14' and '2.0' <= elem[-1] <= '3.0' and elem[0] == 'w':
            lin_wg.append((norma(elem[1]), elem[-1]))
    for elem in nivell1:
        if elem[-2] == '0.14' and elem[-1] == '2.54' and elem[0] == 'y':
            lvl1_ya.append((norma(elem[1]), elem[-2]))
            lvl1_yg.append((norma(elem[1]), elem[-1]))
        elif elem[-2] == '0.14' and elem[-1] == '2.54' and elem[0] == 'w':
            lvl1_wa.append((norma(elem[1]), elem[-2]))
            lvl1_wg.append((norma(elem[1]), elem[-1]))
        elif '0.1' <= elem[-2] <= '0.2' and elem[-1] == '2.54' and elem[0] == 'y':
            lvl1_ya.append((norma(elem[1]), elem[-2]))
        elif '0.1' <= elem[-2] <= '0.2' and elem[-1] == '2.54' and elem[0] == 'w':
            lvl1_wa.append((norma(elem[1]), elem[-2]))
        elif elem[-2] == '0.14' and '2.0' <= elem[-1] <= '3.0' and elem[0] == 'y':
            lvl1_yg.append((norma(elem[1]), elem[-1]))
        elif elem[-2] == '0.14' and '2.0' <= elem[-1] <= '3.0' and elem[0] == 'w':
            lvl1_wg.append((norma(elem[1]), elem[-1]))
    for elem in nivell2:
        if elem[-2] == '0.14' and elem[-1] == '2.54' and elem[0] == 'y':
            lvl2_ya.append((norma(elem[1]), elem[-2]))
            lvl2_yg.append((norma(elem[1]), elem[-1]))
        elif elem[-2] == '0.14' and elem[-1] == '2.54' and elem[0] == 'w':
            lvl2_wa.append((norma(elem[1]), elem[-2]))
            lvl2_wg.append((norma(elem[1]), elem[-1]))
        elif '0.1' <= elem[-2] <= '0.2' and elem[-1] == '2.54' and elem[0] == 'y':
            lvl2_ya.append((norma(elem[1]), elem[-2]))
        elif '0.1' <= elem[-2] <= '0.2' and elem[-1] == '2.54' and elem[0] == 'w':
            lvl2_wa.append((norma(elem[1]), elem[-2]))
        elif elem[-2] == '0.14' and '2.0' <= elem[-1] <= '3.0' and elem[0] == 'y':
            lvl2_yg.append((norma(elem[1]), elem[-1]))
        elif elem[-2] == '0.14' and '2.0' <= elem[-1] <= '3.0' and elem[0] == 'w':
            lvl2_wg.append((norma(elem[1]), elem[-1]))
    for elem in nivell3:
        if elem[-2] == '0.14' and elem[-1] == '2.54' and elem[0] == 'y':
            lvl3_ya.append((norma(elem[1]), elem[-2]))
            lvl3_yg.append((norma(elem[1]), elem[-1]))
        elif elem[-2] == '0.14' and elem[-1] == '2.54' and elem[0] == 'w':
            lvl3_wa.append((norma(elem[1]), elem[-2]))
            lvl3_wg.append((norma(elem[1]), elem[-1]))
        elif '0.1' <= elem[-2] <= '0.2' and elem[-1] == '2.54' and elem[0] == 'y':
            lvl3_ya.append((norma(elem[1]), elem[-2]))
        elif '0.1' <= elem[-2] <= '0.2' and elem[-1] == '2.54' and elem[0] == 'w':
            lvl3_wa.append((norma(elem[1]), elem[-2]))
        elif elem[-2] == '0.14' and '2.0' <= elem[-1] <= '3.0' and elem[0] == 'y':
            lvl3_yg.append((norma(elem[1]), elem[-1]))
        elif elem[-2] == '0.14' and '2.0' <= elem[-1] <= '3.0' and elem[0] == 'w':
            lvl3_wg.append((norma(elem[1]), elem[-1]))

    lin_ya = sorted(lin_ya, key = lambda x: x[1])
    lvl1_ya=sorted(lvl1_ya, key = lambda x: x[1])
    lvl2_ya=sorted(lvl2_ya, key = lambda x: x[1])
    lvl3_ya=sorted(lvl3_ya, key = lambda x: x[1])
    lin_wa = sorted(lin_wa, key = lambda x: x[1])
    lvl1_wa=sorted(lvl1_wa, key = lambda x: x[1])
    lvl2_wa=sorted(lvl2_wa, key = lambda x: x[1])
    lvl3_wa=sorted(lvl3_wa, key = lambda x: x[1])

    lin_yg = sorted(lin_yg, key = lambda x: x[1])
    lvl1_yg=sorted(lvl1_yg, key = lambda x: x[1])
    lvl2_yg=sorted(lvl2_yg, key = lambda x: x[1])
    lvl3_yg=sorted(lvl3_yg, key = lambda x: x[1])
    lin_wg = sorted(lin_wg, key = lambda x: x[1])
    lvl1_wg=sorted(lvl1_wg, key = lambda x: x[1])
    lvl2_wg=sorted(lvl2_wg, key = lambda x: x[1])
    lvl3_wg=sorted(lvl3_wg, key = lambda x: x[1])
    
    
    a = ["{:.3f}".format(float(elem[1])) for elem in lin_ya]
    y1 = [elem[0] for elem in lin_ya]
    y2 = [elem[0] for elem in lvl1_ya]
    y3 = [elem[0] for elem in lvl2_ya]
    y4 = [elem[0] for elem in lvl3_ya]
    w1 = [elem[0] for elem in lin_wa]
    w2 = [elem[0] for elem in lvl1_wa]
    w3 = [elem[0] for elem in lvl2_wa]
    w4 = [elem[0] for elem in lvl3_wa]
    
    g = ["{:.3f}".format(float(elem[1])) for elem in lin_yg]
    y5 = [elem[0] for elem in lin_yg]
    y6 = [elem[0] for elem in lvl1_yg]
    y7 = [elem[0] for elem in lvl2_yg]
    y8 = [elem[0] for elem in lvl3_yg]
    w5 = [elem[0] for elem in lin_wg]
    w6 = [elem[0] for elem in lvl1_wg]
    w7 = [elem[0] for elem in lvl2_wg]
    w8 = [elem[0] for elem in lvl3_wg]

    fig = plt.figure()
    plt.subplot(221)
    plt.title('Sensibilidad y_a') #canviar el gruix de les linies amb lw=num
    plt.plot(a, y1, color='green', marker='.', linestyle='-')#lineal
    plt.plot(a, y2, color='orange', marker='.', linestyle='-')#nivell1
    plt.plot(a, y3, color='blue', marker='.', linestyle='-')#nivell2
    plt.plot(a, y4, color='red', marker='.', linestyle='-')#nivell3
    plt.axvline("{:.3f}".format(0.14), 0, 1)
    plt.xlabel('as')
    plt.ylabel('Norma pesos yn+1')
    plt.grid(True)
    plt.xticks(rotation=70)
    
    plt.subplot(222)
    plt.title('Sensibilidad w_a')
    plt.plot(a, w1, color='green', marker='.', linestyle='-')#lineal
    plt.plot(a, w2, color='orange', marker='.', linestyle='-')#nivell1
    plt.plot(a, w3, color='blue', marker='.', linestyle='-')#nivell2
    plt.plot(a, w4, color='red', marker='.', linestyle='-')#nivell3
    plt.axvline("{:.3f}".format(0.14), 0, 1)
    plt.xlabel('as')
    plt.ylabel('Norma pesos wn+1')
    plt.grid(True)
    plt.xticks(rotation=70)

    plt.subplot(223)
    plt.title('Sensibilidad y_gamma')
    plt.plot(g, y5, color='green', marker='.', linestyle='-')#lineal
    plt.plot(g, y6, color='orange', marker='.', linestyle='-')#nivell1
    plt.plot(g, y7, color='blue', marker='.', linestyle='-')#nivell2
    plt.plot(g, y8, color='red', marker='.', linestyle='-')#nivell3
    plt.axvline("{:.3f}".format(2.54), 0, 1)
    plt.xlabel('gs')
    plt.ylabel('Normas de pesos yn+1')
    plt.grid(True)
    plt.xticks(rotation=70)
    
    ax = plt.subplot(224)
    plt.title('Sensibilidad w_gamma')
    plt.plot(g, w5, color='green', marker='.', linestyle='-')#lineal
    plt.plot(g, w6, color='orange', marker='.', linestyle='-')#nivell1
    plt.plot(g, w7, color='blue', marker='.', linestyle='-')#nivell2
    plt.plot(g, w8, color='red', marker='.', linestyle='-')#nivell3
    plt.axvline("{:.3f}".format(2.54), 0, 1)
    plt.xlabel('gs')
    plt.ylabel('Normas de pesos wn+1')
    plt.grid(True)
    plt.xticks(rotation=70)

    legend_elements = [Line2D([0], [0], color='green', label='lineal', lw=1),
                       Line2D([0], [0], color='orange', label='nivel 1'),
                       Line2D([0], [0], color='blue', label='nivel 2'),
                       Line2D([0], [0], color='red', label='nivel 3')]

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.legend(handles=legend_elements, loc=1, bbox_to_anchor=(1.5, 1.5))
    #loc = 1 especifica la localitzacio i fa qu eno es mogui la llegenda amb el gràfic.

    plt.subplots_adjust(top=0.9, bottom=0.15, left=0.1, right=0.83, hspace=0.5, wspace=0.35)
    
    plt.show()
