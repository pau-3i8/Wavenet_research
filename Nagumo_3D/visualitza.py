import Wavenet_3D as Wavenet
import Nagumo_3D as Nagumo

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
