import Nagumo
import numpy as np
from hopf import limits

def constants(param):
    #parteixo de unes I_min i I_max inicials per calcular les constants per la resta de càlculs
    Iosc1, Iosc2 = limits(param) #amb primera a i gamma
    K1 = (Iosc1-param['I_min'])/(Iosc2-Iosc1)
    K2 = (param['I_max']-Iosc1)/(Iosc2-Iosc1)
    return K1, K2

def I_noves(param, K1, K2):
    #Desplaçaments del rang de Iapps
    Iosc1, Iosc2 = limits(param) #amb noves a i gamma
    param['I_min']=Iosc1-K1*(Iosc2-Iosc1)
    param['I_max']=Iosc1+K2*(Iosc2-Iosc1)
    return param['I_min'], param['I_max']

def log_Is_limit(param, configs):
    K1, K2 = constants(param)
    Is_limit=[]
    for config in configs:
        param['a']= config[0]
        param['gamma']= config[1]
        Is_limit.append((I_noves(param, K1, K2), (param['a'],param['gamma'])))
    f = open('Is_limit.txt', 'w')
    for elem in Is_limit: #arxiu on el primer element de cada columna és la (I_min, I_max) noves
        linia = [str(elem[0])+' '+str(elem[1])+'\n']
        f.writelines(linia)
    f.close()

def sensibilitat(param, configs):
    f = open('Is_limit.txt', 'r')
    weights = []
    progres=0
    for line in f:
        line = line.replace('(', '').replace(')', '').replace(',', '')
        line = line.split()
        param['I_min'] = float(line[0])
        param['I_max'] = float(line[1])
        param['a'] = float(line[2])
        param['gamma'] = float(line[3])
        #aqui generaré els pesos de totes les WN amb les diferents a, gamma i rang de Iapps.
        param['beta']=param['alfa']*param['gamma'] #beta
        #generate=True per generar random Iapps amb els nous rangs
        weights_y, weights_w = Nagumo.approximation(param, param['generateIapp'])
        weights.append(['y', weights_y.T, param['a'], param['gamma']])#per segregar
        weights.append(['w', weights_w.T, param['a'], param['gamma']])#per segregar
        progres += 1
        print('Progrés: '+str(round(progres/len(configs)*100, 2))+'%')
        
    f.close()
    return weights

def log_weights(l):
    f = open('sensibilitat.txt', 'w')
    for elem in l:
        f.write(elem[0]+' ')
        for e in elem[1][0]:
            f.write(str(e)+' ')
        f.writelines([str(elem[2])+' '+str(elem[3])+'\n'])
    f.close()
