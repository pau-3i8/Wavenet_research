import multiprocessing as mp, pandas as pd, numpy as np, psutil, h5py, os, shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dask.distributed import performance_report
from dask import dataframe as dd, array as da
from itertools import product, combinations
from activation_functions import phi, psi
from scipy.linalg import lstsq as lstsq_
from dask.distributed import progress
from numpy.linalg import lstsq
from scipy.linalg import eigh
from numba import njit, types
from numba.typed import Dict
from tqdm import tqdm

# Diccionario con los ratios de compresión según el nivel de resolución. Para evitar problemas con la memoria disponible al exprimir la red. Se ha calculado tomando el peor ratio entre compressiones de 6 a 9 a lo largo de los modelos 2D y 3D.
ratio_comp = {-1:2.5, 0:3, 1:5, 2:13, 3:40}

###################################### DIRECTORY MANAGEMENT ######################################

def set_files(param):
    
    if not param['recovery']:
        main_path = os.getcwd() #t'agafa la direccio del directori al que estàs
        path = os.path.join(main_path, param['matrices_folder']) #path complet del directori /temporary_dir
        if os.path.exists(path): shutil.rmtree(path) #elimines el directori /temporary_dir per fer espai
        os.mkdir(path) #crees el directori /temporary_dir de nou
        os.mkdir(os.path.join(main_path, param['matrix_folder'])) #path complet del directori /FX
        
    if not (param['only_simulation']  or param['recovery_var'] != 0):
        main_path = os.getcwd()
        path = os.path.join(main_path, param['results_folder'])
        if os.path.exists(path): shutil.rmtree(path)
        os.mkdir(path)

###################################### FNCIONES AUXILIARES ########################################

# Selector de matriz
def matriu_Fx(param, inputs):
    if len(inputs) == 3:
        input_1, input_2, Iapp = inputs
        return matriu_2D(param, input_1, input_2, Iapp)
    if len(inputs) == 4:
        input_1, input_2, input_3, Iapp = inputs
        return matriu_3D(param, input_1, input_2, input_3, Iapp)

# Cálculo de wavelons totales
def hidden_layer(param, wn_dimension): #wn_dimension = 2 si 2D i 3 si 3D
    if param['bool_scale']: wavelons = 1
    else: wavelons = 0
    if wn_dimension == 2:
        if param['bool_lineal']: l = 2
        else: l = 0
        for m in range(param['resolution']+1):
            wavelons += (2**(3*m)+3*2**(2*m)+3*2**m)
        neuronios = l+(param['n_sf']**3)*wavelons #neuronas en brasileiro xD
        if (neuronios**2)*param[param['dtype']]/2**30 > psutil.virtual_memory().total/(1024**3):
            print("- ERROR - the WN's level of resolution and number of superposed functions too high, covariance matrix won't fit in memory. Unable to allocate", round((neuronios**2)*param[param['dtype']]/2**30, 2), 'GB')
            exit()
        return neuronios
    #aunque se podria resolver un sistema in memory sin matriz de covariancia con el mimso de neuronas que un sistema out-of-core con una FTF demasiado grande para la RAM, seria con muy pocas filas, es decir, pocas Iapps, porque las frecuencias (integraciones) dependen del sistema de equacions y acostumbran a ser grandes. Para simular un sistema con 5-10 Iapp con niveles altos de resolución no haré casos y directamente corto la ejecución par que se cambie la configuración.
    if wn_dimension == 3:
        if param['bool_lineal']: l = 3
        else: l = 0
        for m in range(param['resolution']+1):
            wavelons += (2**(4*m)+4*2**(3*m)+6*2**(2*m)+4*2**m)
        neuronios = l+(param['n_sf']**4)*wavelons
        if (neuronios**2)*param[param['dtype']]/2**30 > psutil.virtual_memory().total/(1024**3):
            print("- ERROR - the WNN's level of resolution and number of superposed functions too high, covariance matrix won't fit in memory. Unable to allocate", round((neuronios**2)*param[param['dtype']]/2**30, 2), 'GB')
            exit()
        return neuronios
    
########################### MATRIZ FUNCIONES DE ACTIVACIÓN 3 ENTRADAS ###########################

def matriu_2D(param, input_1, input_2, Iapp):
    sf_name = param['fscale']
    n_sf = param['n_sf']
    dtype = param['dtype']
    N = len(input_1)

    matriu = np.zeros((N, hidden_layer(param, 2)), dtype = str(dtype)).T
        
    ## Creas las columnas de la parte lineal
    i = 0
    if param['bool_lineal']:
        i = 2
        #el type de variable de numba es sin string
        #numpy sólo acepta type en formato string
        matriu[0] = np.ones((1, N), dtype = str(dtype))*input_1
        matriu[1] = np.ones((1, N), dtype = str(dtype))*input_2
    
    ## Creas las columnas de funciones de escala
    if param['bool_scale']:
        n = [n for n in range(n_sf)]
        for ns in list(product(n,n,n)):
            n1, n2, n3 = ns
            matriu[i] = phi(sf_name, input_1, n1, dtype)* phi(sf_name, input_2, n2, dtype)*phi(sf_name, Iapp, n3, dtype)
            i+=1

    ## Creas las columnas de wavelets
    for m in range(param['resolution']+1):
        n = [n for n in range(n_sf)]
        c = [c for c in range(2**m)]
        v = [(input_1, input_2, Iapp), (Iapp, input_1, input_2), (input_2, Iapp, input_1)]

        #Les K's corresponen als factors de l'equacio 6.11 del TFG amb els valors de la taula 6.2
        for elem in list(product(n,n,n)):
            n1, n2, n3 = elem
            for var in v:
                for c1 in c: #K3
                    matriu[i] = phi(sf_name, var[0], n1, dtype)* phi(sf_name, var[1], n2, dtype)* psi(sf_name, (2**m)* var[2] - c1, n3, dtype)
                    i+=1
                for ci in list(product(c,c)): #K2
                    c1, c2 = ci
                    matriu[i] = phi(sf_name, var[0], n1, dtype)* psi(sf_name, (2**m)* var[1] - c1, n2, dtype)* psi(sf_name, (2**m)* var[2] - c2, n3, dtype)
                    i+=1
            for ci in list(product(c,c,c)): #K1
                c1, c2, c3 = ci
                matriu[i] = psi(sf_name, (2**m)* input_1 - c1, n1, dtype)* psi(sf_name, (2**m)* input_2 - c2, n2, dtype)* psi(sf_name, (2**m)* Iapp - c3, n3, dtype)
                i+=1
                
    return matriu.T


########################### MATRIZ FUNCIONES DE ACTIVACIÓN 4 ENTRADAS ###########################

def matriu_3D(param, input_1, input_2, input_3, Iapp):
    sf_name = param['fscale']
    n_sf = param['n_sf']
    dtype = param['dtype']
    N = len(input_1)
    
    matriu = np.zeros((N, hidden_layer(param, 3)), dtype = str(dtype)).T   

    ## Creas las columnas de la parte lineal
    i = 0
    if param['bool_lineal']:
        i = 3
        matriu[0] = np.ones((1, N), dtype = str(dtype))*input_1
        matriu[1] = np.ones((1, N), dtype = str(dtype))*input_2
        matriu[2] = np.ones((1, N), dtype = str(dtype))*input_3
        
    ## Creas las columnas de funciones de escala
    if param['bool_scale']:
        n = [n for n in range(n_sf)]
        for ns in list(product(n,n,n,n)):
            n1, n2, n3, n4 = ns
            matriu[i] = phi(sf_name, input_1, n1, dtype)* phi(sf_name, input_2, n2, dtype)* phi(sf_name, input_3, n3, dtype)* phi(sf_name, Iapp, n4, dtype)
            i+=1

    ## Creas las columnas de wavelets
    for m in range(param['resolution']+1):
        n = [n for n in range(n_sf)]
        c = [c for c in range(2**m)]
        v1 = [(input_2, input_1, input_3, Iapp), (Iapp, input_2, input_1, input_3), (input_3, Iapp, input_2, input_1), (input_1, input_3, Iapp, input_2)]
        v2 = [(input_2, input_1, input_3, Iapp), (Iapp, input_2, input_1, input_3), (input_3, Iapp, input_2, input_1), (input_1, input_3, Iapp, input_2), (input_3, input_2, input_1, Iapp), (input_1, Iapp, input_3, input_2)]

        #Les K's corresponen als factors de l'equacio 6.11 del TFG amb els valors de la taula 6.2
        for ns in list(product(n,n,n,n)):
            n1, n2, n3, n4 = ns
            for var in v1: #K4
                for c1 in c:
                    matriu[i] = phi(sf_name, var[0], n1, dtype)* phi(sf_name, var[1], n2, dtype)* phi(sf_name, var[2], n3, dtype)* psi(sf_name, (2**m)* var[3] - c1, n4, dtype)
                    i+=1
            for var in v2: #K3
                for ci in list(product(c,c)):                
                    c1, c2 = ci
                    matriu[i] = phi(sf_name, var[0], n1, dtype)* phi(sf_name, var[1], n2, dtype)* psi(sf_name, (2**m)* var[2] - c1, n3, dtype)* psi(sf_name, (2**m)* var[3] - c2, n4, dtype)
                    i+=1
            for var in v1: #K2
                for ci in list(product(c,c,c)):
                    c1, c2, c3 = ci
                    matriu[i] = psi(sf_name, (2**m)* var[0] - c1, n1, dtype)* psi(sf_name, (2**m)* var[1] - c2, n2, dtype)* psi(sf_name, (2**m)* var[2] - c3, n3, dtype)* phi(sf_name, var[3], n4, dtype)
                    i+=1
            for ci in list(product(c,c,c,c)): #K1
                c1, c2, c3, c4 = ci
                matriu[i] = psi(sf_name, (2**m)* input_2 - c1, n1, dtype)* psi(sf_name, (2**m)* input_1 - c2, n2, dtype)* psi(sf_name, (2**m)* input_3 - c3, n3, dtype)* psi(sf_name, (2**m)* Iapp - c4, n4, dtype)
                i+=1
    
    return matriu.T

################################## FUNCIONES PARA GENERAR DATOS ##################################
    
### CREATE IAPPS
def generate_data(param, n_Iapps, seeds):
    Iapps = np.linspace(param['I_min'], param['I_max'], n_Iapps)
    for s in seeds:
        np.random.seed(s)
        np.random.shuffle(Iapps)
    return Iapps #és un array amb les Iapps sense repetir n_vegades cada Iapp pel nº integracions

### IMPORT IAPPS FROM A FILE
def import_data(param):
    Iapps = read_data('inputs.parquet')
    return Iapps

## Serveix per guardar Iapps i outputs de una sola dimensió
def save_data(arxiu, vector):
    #vector es de variable de tipo dask.array
    dd.from_dask_array(vector, columns = ['0']).to_parquet(arxiu)
    
## Serveix per llegir Iapps i outputs de una sola dimensió
def read_data(arxiu):
    #estructura datos archivo = [[elem 1],[elem 2],[...],[elem n]]
    #retorn np.array([n elem])
    return np.array(pd.read_parquet(arxiu)).T[0]

####################################### IN MEMORY ALGORITHM #######################################
from hyperlearn.solvers import solve

### APPROXIMATION
def in_memory_approx(param, var, Iapps, tuples):
    inputs = tuple([input_[0] for input_ in tuples])+(Iapps,)
    Fx=  matriu_Fx(param, inputs)
    print('')
    for i in range(len(tuples)):
        print('--- Aproximando', var[i], '---')
        target = tuples[i][1].reshape(len(tuples[i][1]), 1) #ajusta la dimensió pel training
        save_data(param['results_folder']+'/weights_' + var[i] + '.parquet', in_memory_training(param, Fx, target, var[i]))
            
### LOSS
def in_memory_training(param, Fx, target, var):
    ## Fastest
    weights = np.array([solve(Fx, target, alpha=param['regularizer'])]).T #vector columna
    """
    FTF = np.dot(Fx.T, Fx)
    vaps = eigh(FTF, eigvals_only=True)
    A = FTF + np.identity(FTF.shape[0])*param['regularizer']*vaps[-1].real
    b = np.dot(Fx.T, target)
    weights = lstsq(A, b, rcond=None)[0].T
    """
    ## Compute approximation error
    print('->', var, 'MSE at level =', param['resolution'] + 2, 'is:', np.sum((target-Fx.dot(weights))**2)/len(target))
    return da.from_array(weights) #paso a dask.arrya para guardarlo con save_data()

###################################### OUT-OF-CORE ALGORITHM ######################################

### APPROXIMATION
# tuples = [(input_1, target_1), (input_2, target_2) [...]] funció genèrica
def bloc_write(i, nf, factor, files_chunk, wavelons, param, Iapps, tuples):
    with h5py.File(param['matrix_folder']+'/matriu'+'0'*(4-len(str(i)))+str(i)+'.hdf5', 'w') as f:
        dset = f.create_dataset('dset', shape = (nf, wavelons), chunks = (nf, wavelons), compression = 'gzip', compression_opts = param['compression'], dtype = str(param['dtype']))
        for j in range(factor):
            inputs = [input_[0][nf*i:nf*(i+1)][files_chunk*j:files_chunk*(j+1)] for input_ in tuples]+[Iapps[nf*i:nf*(i+1)][files_chunk*j:files_chunk*(j+1)]]
            dset[files_chunk*j:files_chunk*(j+1)] = matriu_Fx(param, inputs)
            
def out_of_core_approx(param, var, Iapps, tuples, files_chunk, n_chunks, wavelons):
    factor, cpu_compensator = compactador(param, n_chunks, files_chunk, wavelons)
    #factor = n_chunks_anteriors/arxiu
    nf = files_chunk*factor #aumento files_chunk per guardar chunks més grans
    n_chunks //= factor # 1 dataset/arxiu || 1 chunk = 1 arxiu
    
    if not param['recovery']:
        with ProcessPoolExecutor(mp.cpu_count()-cpu_compensator) as ex:
            futures = [ex.submit(bloc_write, i, nf, factor, files_chunk, wavelons, param, Iapps, tuples) for i in range(n_chunks)]
            [_ for _ in tqdm(as_completed(futures), total=len(futures), desc='Guardant matriu creada', unit='arxiu', leave=True)]
    
    arxius_oberts=[]
    with performance_report(): #genera dask-report.html
        datasets = []
        try:
            for i in tqdm(range(n_chunks), desc='Llegint matriu creada', leave=False):
                f = h5py.File(param['matrix_folder']+'/matriu'+'0'*(4-len(str(i)))+str(i)+'.hdf5', 'r')
                arxius_oberts.append(f)
                datasets.append(da.from_array(f.get('dset'), chunks=(files_chunk, wavelons)))
                # cada dataset = nuevo_chunk corresponde al tamaño de todo un archivo, y se subdivide en chunk del tamaño acorde a las fitas definidas. No se pude generar un archivo con el tamaño del chunk igual al del archivo, por la variable factor > 1, si no la simulación se para (por culpa del exit() del final de la funcion compactador())
            FX = da.concatenate(datasets, axis=0)
            print('Guardant matriu de covariancia FTF')
            if not param['recovery_FTF']:
                FTF = dd.from_dask_array(da.dot(FX.T, FX), columns = [str(elem) for elem in np.arange(wavelons)])
                FTF = FTF.persist() #he d'executar els calculs al background per veure el progres
                progress(FTF) #per veure la barra de progress amb distributed.
                FTF.to_parquet(param['matrices_folder']+'/FTF.parquet')
            for i in range(param['recovery_var'],len(tuples)):
                print('\n'+'--- Aproximando', var[i], '---')
                target = tuples[i][1].reshape(len(tuples[i][1]), 1)
                out_of_core_training(param, FX, target, files_chunk, var[i])
        finally:
            #el finally tanca tots els arxius oberts encara que apareguin errors.
            [arxiu.close() for arxiu in arxius_oberts]
    print('')
                                
### LOSS
def out_of_core_training(param, FX, target, files_chunk, var):
    FTF = np.array(pd.read_parquet(param['matrices_folder']+'/FTF.parquet'))
    vaps = eigh(FTF, eigvals_only=True)
    #per matriu més grans de 2¹⁵ amb eigh() de numpy peta. pero amb scipy no
    A = FTF + np.identity(FTF.shape[1])*param['regularizer']*vaps[-1].real
    
    Y = da.from_array(target, chunks = (files_chunk, 1))
    #dd.from_dask_array(da.dot(FX.T, Y), columns = ['0']).to_parquet('FTY.parquet')
    FTY = dd.from_dask_array(da.dot(FX.T, Y), columns = ['0'])
    FTY = FTY.persist()
    progress(FTY)
    FTY.to_parquet(param['matrices_folder']+'/FTY.parquet')
    b = np.array(pd.read_parquet(param['matrices_folder']+'/FTY.parquet'))

    weights = lstsq(A, b, rcond=None)[0] #vector columna
    #weights = lstsq_(A, b, cond=None)[0] #vector columna amb scipy
    weights = da.from_array(weights, chunks = (files_chunk, 1))
    save_data(param['results_folder']+'/weights_' + var + '.parquet', weights)
    
    #Si m'interessa el MSE descomento la següent línia però és fer perdre temps.
    #print('->', var, 'MSE at level =', param['resolution']+2, 'is:', out_of_core_MSE(FX, target, weights))

def out_of_core_MSE(FX, target, weights):
    save_data(param['matrix_folder']+'/temp.parquet', da.dot(FX, weights))   
    FW = read_data(param['matrices_folder']+'/temp.parquet')
    #l'arxiu temp és F(x)*w. Només el vull per poder convertir-ho en un array de poca memoria.
    return np.sum((target - FW)**2)/len(target)

################################# FUNCIONES COMUNES DEL ALGORITMO #################################
    
### SIMULATION
# tuples = [(ci_1, target_1), (ci_2, target_2) [...]] funció genèrica
def simulation(param, var, Iapps, tuples): #introdueixo una funcio func com a argument
    d_var={}
    for i,elem in enumerate(tuples):
        d_var['weights_'+str(i)] = np.array(pd.read_parquet(param['results_folder']+'/weights_' + var[i] + '.parquet')) #no hago read_data pq quiero que el vector tenga 2 dimensiones
        d_var['t_'+str(i)] = elem[1]
        d_var['i_'+str(i)] = np.array([elem[0]])
        d_var['predicted_'+str(i)] = np.zeros_like(elem[1])

    for j,I in enumerate(tqdm(Iapps, desc='Predicció', unit=' integracions', leave=False)):
        normalized = normalize(param, (d_var['i_'+str(i)] for i in range(len(tuples))), np.array([I]))
        Fx = matriu_Fx(param, normalized)
        for i in range(len(tuples)):
            d_var['i_'+str(i)] = (Fx.dot(d_var['weights_'+str(i)]))[0] #n+1
            d_var['predicted_'+str(i)][j] = d_var['i_'+str(i)] #guardo la prediccio
            
    for i in range(len(tuples)): #resultats
        print('->', var[i],'RMSE:', np.sum((d_var['t_'+str(i)]-d_var['predicted_'+str(i)])**2)/np.sum((np.mean(d_var['t_'+str(i)])-d_var['t_'+str(i)])**2)*100,'%')
        save_data(param['results_folder']+'/predicted_' + var[i] + '.parquet', da.from_array(d_var['predicted_'+str(i)]))

def domain_limits(param, euler_dict, func, **cond_ini_approx):
    # No hago np.random.uniform para ahorrarme el shufle precisament porue así puedo tener un numero de Is con valores equidistantes y luego solo tengo quehacer un shuffle. para la aproximación hago lo mismo y así la red puede interpolar de manera más uniforme en todo el domino para qualquier tipo de Iapp generada en la simulación, donde sí quegenero valores uniformemente distribuidos.
    Is = np.linspace(param['I_min']-param['domain_margin']*abs(param['I_min']),
                     param['I_max']+param['domain_margin']*abs(param['I_max']),
                     10000)
    """10000 Iapps per fer un bon barrido i +-param['domain_margin'] per aumentar el domini i evitar errors al training, per normalitzar dades a prop dels limits dle domini."""
    np.random.seed(1)
    np.random.shuffle(Is)
    
    euler_dict['points'] = param['points']
    #aqui no importa que no sea con los puntos finales, sólo calculo los limites.
    #per tenir sempre el mateix sampling per trobar max i min.
    _, targets_euler = func(euler_dict, Is, **cond_ini_approx)
    for i,elem in enumerate(targets_euler):
        param['max'+str(i)] = np.max(elem)
        param['min'+str(i)] = np.min(elem)
        
### NORMALITZATION
def normalize(param, train_data, Iapp): #train_data = (w_train, y_train, ...)
    dnorm = {}
    for i,data in enumerate(train_data):
        dnorm['norm'+str(i)] = (data-param['min'+str(i)])/(param['max'+str(i)]-param['min'+str(i)])
    Inorm = (Iapp-param['I_min'])/(param['I_max']-param['I_min'])
    return tuple([norm for norm in dnorm.values()])+(Inorm,)

####################################### WN PARAMETRIZATION #######################################

## Selecciona el algoritmo (in memory o out-of-core)
def select_algorithm(param, wavelons, marker = 0):
    memoria = param['points']*param['n_Iapp']*wavelons*param[param['dtype']]/(2**30)
    #si no se usara hyperlearn el factor_ser tendria que ser 2.6 - 2.9
    if memoria <= psutil.virtual_memory().free/(1024**3):
        mode  = 'IN MEMORY MODE'
        n_chunks, files_chunk, mode = in_memory_postconfig(param, wavelons, mode)
    elif (memoria <= psutil.disk_usage('/')[2]/(1024**3)*ratio_comp[param['resolution']]/1.1) or param['recovery']:
    # se supone que si recovery = True ya hay una carpeta FX donde importar la matriz.
    #1.1 para dar un margen del 10% de espacio al disco para el directorio temporal de distributed
        mode  = 'OUT-OF-CORE MODE'
        n_chunks, files_chunk, mode = parametrization(param, wavelons, marker, mode)
    else:
        print('\n','- ERROR - No hay suficiente memoria para la configuración de la Wavenet.')
        print('Available disk space:', round(psutil.disk_usage('/')[2]/(2**30), 2), 'GB', 'Required disk space:', round(memoria/ratio_comp[param['resolution']], 2), 'GB')
        pass#exit()
    return n_chunks, files_chunk, mode

# troba el núm files_chunk més gran (el mín n_chunks) entre les fites donades.
def chunk_size(param, wavelons, marker):
    matrix_MB = param['points']*param['n_Iapp']*wavelons*param[param['dtype']]/(2**20)
    n_chunks_max = int(matrix_MB/param['fita_chunk_inf']+1) #+1 pq al hacer int redondee arriba
    # n_chunks_max corresponde a al n_chunks maxim, de min n_filas
    interruptor = 0
    while n_chunks_max > matrix_MB/param['fita_chunk_sup']:
        files_chunk = param['points']*param['n_Iapp']/n_chunks_max
        if files_chunk % 1 == 0 and matrix_MB/n_chunks_max > param['fita_chunk_inf']:
            #si no té decimals y por encima de la fita inf
            interruptor = 1
            break
        n_chunks_max -= 1
    if interruptor == 1: return n_chunks_max #n_chunks valido dentro de las fitas
    else:
        print('- ERROR - No hay una configuración possible.')
        if marker == 0:
            new_Iapps = input('Enter new n_Iapp value: ')
            param['n_Iapp'] = int(new_Iapps) #sobrescribo param
            return chunk_size(param, wavelons, marker)
        elif marker == 1:
            new_punts = input('Enter new n_points value: ')
            param['points'] = int(new_punts)
            return chunk_size(param, wavelons, marker)
    
def in_memory_postconfig(param, wavelons, mode):
    print('')
    print(display_configuration(param, wavelons, 1, mode))
    print('Change n_Iapps [i], nº of points [p]?')
    answer = input("(i/p): ")
    if answer == 'i':
        new_Iapps = input('Enter new n_Iapp value: ')
        param['n_Iapp'] = int(new_Iapps) #sobrescribo param
        print('\n'*100)
        return select_algorithm(param, wavelons)
    #como marker tiene un valor default y aqui no se usa no hace falta importarlo a la funcion
    elif answer == 'p':
        new_punts = input('Enter new n_points value: ')
        param['points'] = int(new_punts)
        print('\n'*100)
        return select_algorithm(param, wavelons)
    return 1, param['n_Iapp']*param['points'], mode
    
def parametrization(param, wavelons, marker, mode):
    #marker es un marcador para saber de qué cambio vienes, si de cambiar las Iapps o el n_puntos.
    n_chunks = chunk_size(param, wavelons, marker)
    #els chunks sense compactar tenen tantes files com files_chunk
    print('') # fake salt de linia perque només es mostri la config. que estic modificant 
    print(display_configuration(param, wavelons, n_chunks, mode))
    print('Change n_Iapps [i], nº of points [p], fita_chunk_inf [inf] or fita_chunk_sup [sup]?')
    answer = input("(i/p/inf/sup): ")
    if answer == 'i':
        new_Iapps = input('Enter new n_Iapp value: ')
        param['n_Iapp'] = int(new_Iapps) #sobrescribo param
        print('\n'*100)
        return select_algorithm(param, wavelons, 0)
    elif answer == 'p':
        #si tienes que arreglar problemas de configuración sólo tocar: n_Iapps o n_puntos
        new_punts = input('Enter new n_points value: ')
        param['points'] = int(new_punts)
        print('\n'*100)
        return select_algorithm(param, wavelons, 1)
    elif answer == 'inf':
        new_fita_inf = input('Enter new fita_inf value [MB]: ')
        param['fita_chunk_inf'] = int(new_fita_inf)
        print('\n'*100)
        return select_algorithm(param, wavelons, marker)
    elif answer == 'sup':
        new_fita_sup = input('Enter new fita_sup value [MB]: ')
        param['fita_chunk_sup'] = int(new_fita_sup)
        print('\n'*100)
        return select_algorithm(param, wavelons, marker)
    print('')
    return n_chunks, param['n_Iapp']*param['points']//n_chunks, mode

# Descomposicio factorial del número de wavelons i parametritzar la matriu en valors multiples
def descomp_factorial(num):
    factor = 2
    l = []
    while factor*factor <= num:
        if num % factor:
            factor += 1
        else:
            num = num//factor
            l.append(factor)
    if num > 1:
        l.append(num)
    return l

def producte(n):
    p = n[0]
    for elem in n[1:]:
        p *= elem
    return p

def divisors(l):
    #ha de calcular tots els possibles divisors, per tant, farà combinacions de n-1 factor
    #si fa combinacions dels n factors directament obtens el valor que havies factoritzat.
    n_factors = np.arange(1, len(l)) #combinacios de 1 a len(l)-1 factors
    ll = []
    for n in n_factors:
        comb_factors = list(combinations(l, n))
        divisors = list(map(producte, comb_factors))
        ll.append(divisors)
    div = list(set([e for elem in ll for e in elem])) #el set() és per eliminar repetits i ORDENA
    return div

def compactador(param, n_chunks, files_chunk, wavelons):
    div = divisors(descomp_factorial(n_chunks))
    div.reverse() #comença pels divisors més grans
    cpu_compensator = 0 #si hace falta reducir el numero de CPUs en parallelo para reducir la RAM utilizada sin sacrificar el computo en parallelo, esta variable lo corrige
    while cpu_compensator < mp.cpu_count(): #hasta agotar el núemro de CPUs disponibles
        for elem in div:
            if n_chunks/elem < 950: #n_chunks = n_arxius i linux soporta fins 1024 arxius simultanis.
                if files_chunk*elem*wavelons*param[param['dtype']]/(2**30) <= psutil.virtual_memory().free/(2**30)/(mp.cpu_count()-cpu_compensator)/param['sec_factor']: #max dask chunk [GB]
                    print('Compressing', n_chunks//elem, 'files at', round(files_chunk*elem*wavelons*param[param['dtype']]/(2**30), 2), 'GB/files')
                    print('Minus', cpu_compensator, 'CPU/s \n')
                    return elem, cpu_compensator
        cpu_compensator += 1
    print('- ERROR - La configuración actual no permite reducir el número de archivos para que Linux pueda gestionarlos todos a la vez.')
    exit()

###################################### FUNCIONES PRINCIPALES ######################################

@njit
def dic(dic):
    jit_dic = Dict.empty(key_type=types.unicode_type,
                         value_type=types.float64)
    
    for key,value in dic.items():
      jit_dic[key] = value
    return jit_dic

def approximation(param, euler_dict, euler, var, **CI_approx):
    ## Auto configuració parametritzada
    wavelons = hidden_layer(param, len(var)) #calcula el nº de wavelons
    # si es in memory sólo habrá 1 chunk = la matriz entera
    n_chunks, files_chunk, mode = select_algorithm(param, wavelons)
    
    seeds = [5061996, 5152017]
    if param['generateIapp']:
        Iapps = generate_data(param, param['n_Iapp'], seeds)
        save_data('inputs.parquet', da.from_array(Iapps))
    else: Iapps = import_data(param) #si ho importo no cal guardar els inputs

    euler_dict['points'] = param['points'] #pq el nº de punto quizás cambia en select_algorithm()
    train_data, target = euler(dic(euler_dict), Iapps, **CI_approx)
    Iapps = np.array([I for I in Iapps for n_times in range(param['points'])])
    #aquí el vector Iapps ha de ser tan gran com el total d'integracions per Iapp
    #inputs = input_1, input_2, ..., Iapps
    inputs = normalize(param, train_data, Iapps) # inputs[-1] = normalized Iapps
    tuples = [(input_, target[i]) for i,input_ in enumerate(inputs[:-1])]
    if mode == 'IN MEMORY MODE':
        in_memory_approx(param, var, inputs[-1], tuples)
    if mode == 'OUT-OF-CORE MODE':
        out_of_core_approx(param, var, inputs[-1], tuples, files_chunk, n_chunks, wavelons)
    
def prediction(param, euler_dict, euler, var, titulo, **CI_simu):
    np.random.seed(80085)
    if param['generateIapp_simu']:
        Iapps = np.random.uniform(param['I_min'], param['I_max'], param['n_Iapp_simu'])
    # aquí si que testeo Iapps totalmente random sin valores equidistantes
    else:
        Iapps = import_data(param)
        if param['shuffle']: np.random.shuffle(Iapps)
        Iapps = Iapps[:param['n_Iapp_simu']] #millor retallar després del shuffle si vull 1 Iapp

    euler_dict['points'] = param['points_simu']
    _, target = euler(dic(euler_dict), Iapps, **CI_simu)
    Iapps = np.array([I for I in Iapps for n_times in range(param['points_simu'])])
    #aquí també el vector Iapps ha de ser tan gran com el total d'integracions per Iapp
    tuples = [(CI, target[i]) for i,CI in enumerate(CI_simu.values())]
    #CI_simu és un diccionari les CI són els valors de cada key
    simulation(param, var, Iapps, tuples)
    #si solo se quieren ver los graficos, los outputs ya deberían estar guardados y solo necesitas generar los targets con la funcion euler()
    
    ## Gràfics
    visualize(param, titulo, var, Iapps, target)

############################################ GRAPHICS ############################################

### Taula de la terminal
from terminaltables import SingleTable

def display_configuration(param, wavelons, n_chunks, mode):
    nbytes = param['points']*param['n_Iapp']*wavelons*param[param['dtype']] #nbytes matrix
    config = [['n_chunks', n_chunks,
               'rows_chunk', param['n_Iapp']*param['points']//n_chunks],
              ['n_points', param['points'],
               'n_Iapps', param['n_Iapp']],
              ['fita_chunk_inf', param['fita_chunk_inf'],
               'fita_chunk_sup', param['fita_chunk_sup']],
              ['Memoria F(x)', str(round(nbytes/(2**30), 2)) + ' GB',
               'Memoria/chunk', str(round(nbytes/n_chunks/(2**20),2)) + ' MB']]
    
    taula = SingleTable(config, ' WAVENET WITH '+ str(wavelons) +' WAVELONS '+'--> '+ mode +' ')
    taula.inner_row_border = True
    taula.justify_columns = {0: 'center', 1: 'center', 2: 'center', 3: 'center'}
    return taula.table

### GRAPHICS
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

#graphic's format
pylab.rcParams['figure.figsize'] = 11, 12
pylab.rcParams['axes.titlesize'] = 20
pylab.rcParams['axes.labelsize'] = 16
pylab.rcParams['xtick.labelsize'] = 14
pylab.rcParams['ytick.labelsize'] = 14
pylab.rcParams['legend.fontsize'] = 14

##Gràfics dels resultats
def visualize(param, titol, var, Iapps, targets):
    d_var={}
    for i,elem in enumerate(targets):
        d_var['pred_'+str(i)] = read_data(param['results_folder']+'/predicted_'+var[i]+'.parquet')
        time_graphic(titol, var[i], Iapps, d_var['pred_'+str(i)], elem)
    # Si el algoritmo fuera != 2D o 3D phase_portarit no funcionaria porque es un grafico 3D
    # tendrian que seleccionar-se las variables a mostrar en cada caso
    if len(var) == 2:
        outputs  = [d_var['pred_'+str(i)] for i in range(len(var))]+[Iapps]
        targets  = targets+(Iapps,)
        var = var+['Iapp']
        # Diagrama de fases 2D
        fig = plt.figure()
        plt.plot(targets[1], targets[0], label='Target', color='blue', linestyle='-', lw = 0.6)
        plt.plot(outputs[1], outputs[0], label='WNN', color='orange', linestyle='-', lw = 0.5)
        plt.xlabel(var[0])
        plt.ylabel(var[1])
        plt.legend()
        plt.savefig('Phase_portrait_2D.png')
        plt.show()
    # Diagrama de fases 3D
    phase_portrait(titol, var, [d_var['pred_'+str(i)] for i in range(len(var))], targets)

def phase_portrait(titol, var, outputs, targets):
    predict_1, predict_2, predict_3 = outputs
    tar_1, tar_2, tar_3 = targets
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title(titol)
    ax.plot(tar_1, tar_3, tar_2, label='Target', color='blue', linestyle='dotted', lw = 0.6)
    ax.plot(predict_1, predict_3, predict_2, label='WNN', color='orange', linestyle='dotted', lw = 0.5)
    ax.set_xlabel(var[0])
    ax.set_ylabel(var[2])
    ax.set_zlabel(var[1])
    ax.legend()    
    plt.show()

def time_graphic(titol, var, Iapps, predicted_data, target):
    
    time=[]
    [time.append(i) for i in range(len(target))]
    
    plt.figure()
    plt.subplot(211)
    plt.title(titol)
    plt.xlabel('Steps')
    plt.ylabel(var)
    plt.plot(time, target, label='Target', color='blue', linestyle='-')
    plt.plot(time, predicted_data, label='WNN', color='orange', linestyle='-')
    plt.legend()

    plt.subplot(212)
    plt.title('Randomised forced term')
    plt.xlabel('Steps')
    plt.ylabel('Iapp')
    plt.plot(time, Iapps, color='blue', marker=',', linestyle='-')
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.10, right=0.95, hspace=0.5, wspace=0.35)
    
    plt.savefig('Outputs_'+var+'.png')
    plt.show()
