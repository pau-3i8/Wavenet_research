import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def grafic_pesos(param):
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

    s = param['n_sf']**3
    segrega = [2, 2+s, (7*s)+(s+2), (s*26)+(7*s)+(s+2)]
    
    x = np.arange(1, len(ya_first)+1, 1)
    fig = plt.figure()
    plt.subplot(221)
    plt.title('Sensibilidad y_a') #canviar el gruix de les linies amb lw=num
    plt.plot(x, ya_ref, color='green', marker=',', linestyle='-')#ref
    plt.plot(x, ya_first, color='orange', marker=',', linestyle='--')#(0.1, 2.54)
    plt.plot(x, ya_last, color='blue', marker=',', linestyle='--')#(0.2, 2.54)
    for elem in ya_i:
        plt.plot(x, elem, color='red', marker=',', linestyle='--')#el montón
    for barra in segrega:
        plt.axvline(barra, 0, 1, lw=1)
    plt.xlabel('Dimensión de la base')
    plt.ylabel('Pesos yn+1')
    plt.grid(True)
    
    plt.subplot(222)
    plt.title('Sensibilidad w_a')
    plt.plot(x, wa_ref, color='green', marker=',', linestyle='-')#ref
    plt.plot(x, wa_first, color='orange', marker=',', linestyle='--')#(0.1, 2.54)
    plt.plot(x, wa_last, color='blue', marker=',', linestyle='--')#(0.2, 2.54)
    for elem in wa_i:
        plt.plot(x, elem, color='red', marker=',', linestyle='--')#el montón
    for barra in segrega:
        plt.axvline(barra, 0, 1, lw=1)
    plt.xlabel('Dimensión de la base')
    plt.ylabel('Pesos wn+1')
    plt.grid(True)

    plt.subplot(223)
    plt.title('Sensibilidad y_gamma')
    plt.plot(x, yg_ref, color='green', marker=',', linestyle='-')#ref
    plt.plot(x, yg_first, color='orange', marker=',', linestyle='-')#(0.14, 2)
    plt.plot(x, yg_last, color='blue', marker=',', linestyle='-')#(0.14, 3)
    for elem in yg_i:
        plt.plot(x, elem, color='red', marker=',', linestyle='-')#el montón
    for barra in segrega:
        plt.axvline(barra, 0, 1, lw=1)
    plt.xlabel('Dimensión de la base')
    plt.ylabel('Pesos yn+1')
    plt.grid(True)
    
    ax = plt.subplot(224)
    plt.title('Sensibilidad w_gamma')
    plt.plot(x, wg_ref, color='green', marker=',', linestyle='-')#ref
    plt.plot(x, wg_first, color='orange', marker=',', linestyle='-')#(0.14, 2)
    plt.plot(x, wg_last, color='blue', marker=',', linestyle='-')#(0.14, 3)
    for elem in wg_i:
        plt.plot(x, elem, color='red', marker=',', linestyle='-')#el montón
    for barra in segrega:
        plt.axvline(barra, 0, 1, lw=1)
    plt.xlabel('Dimensión de la base')
    plt.ylabel('Pesos wn+1')
    plt.grid(True)

    legend_elements = [Line2D([0], [0], color='green', label='referencia', lw=1),
                       Line2D([0], [0], color='red', linestyle='--', label='diferentes as'),
                       Line2D([0], [0], color='blue', linestyle='--', label='a=0.2 gamma=2.54'),
                       Line2D([0], [0], color='orange', linestyle='--', label='a=0.1 gamma=2.54'),
                       Line2D([0], [0], color='red', label='diferentes gammas', lw=1),
                       Line2D([0], [0], color='blue', label='a=0.14 gamma=3.0', lw=1),
                       Line2D([0], [0], color='orange', label='a=0.14 gamma=2.0', lw=1)]

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.legend(handles=legend_elements, loc=1, bbox_to_anchor=(1.65, 1.65))
    #loc = 1 especifica la localitzacio i fa qu eno es mogui la llegenda amb el gràfic.

    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.10, right=0.8, hspace=0.5, wspace=0.35)
    
    plt.show()
