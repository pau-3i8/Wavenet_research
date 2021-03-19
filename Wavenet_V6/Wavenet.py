from dask.distributed import performance_report, Client, LocalCluster
import pandas as pd, numpy as np, psutil, os, shutil, zarr, dask
from activation_functions import matriu_3D, matriu_2D 
from dask import dataframe as dd, array as da
from itertools import product, combinations
from dask.distributed import progress
from numpy.linalg import lstsq
from scipy.linalg import eigh
from numba import njit, types
from numba.typed import Dict
from numcodecs import Blosc
from tqdm import tqdm

# Diccionario con los ratios de compresión según el nivel de resolución. Para evitar problemas con la memoria disponible al exprimir la red. Se ha calculado tomando el peor ratio entre compressiones de 1 a 6 a lo largo de los modelos 2D y 3D.
ratio_comp = {-1:3, 0:3, 1:6, 2:15, 3:30}

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
        return matriu_2D(param, input_1, input_2, Iapp, hidden_layer(param, 2))
    #no hace falta modificar los datos creados en f8 a f4 con .astype('float32') porque ya se modifican al llenar la matriz FX i se hacen a partir de los funciones de activación ya operadas con 16 decimales (f8), permitiendo mejorar el error, a diferencia de si se modificaran los datos aquí antes de operar.
    if len(inputs) == 4:
        input_1, input_2, input_3, Iapp = inputs
        return matriu_3D(param, input_1, input_2, input_3, Iapp, hidden_layer(param, 3))

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

###################################### OUT-OF-CORE ALGORITHM ######################################

### APPROXIMATION
# tuples = [(input_1, target_1), (input_2, target_2) [...]] funció genèrica
def out_of_core_approx(param, var, Iapps, tuples, files_chunk, n_chunks, wavelons):
    factor = compactador(param, n_chunks, files_chunk, wavelons)
    #factor = n_chunks_anteriors/n_chunks_para_la_escritura_de_la_matrix
    nf = files_chunk*factor #aumento files_chunk per guardar chunks més grans
    n_chunks //= factor # 1 dataset/chunk || 
    
    if not param['recovery']:
        ## SE GENERA LA MATRIX EN ARCHIVOS .ZARR ##
        compressor = Blosc(cname=param['cname'], clevel=param['clevel'])#, shuffle=Blosc.SHUFFLE)
        synchronizer = zarr.ProcessSynchronizer('temp.sync')
        f = zarr.DirectoryStore(param['matrix_folder']+'/matriu.zarr')
        z = zarr.create(store = f, shape=(Iapps.shape[0], wavelons),
                        overwrite = True, compressor = compressor,
                        synchronizer = synchronizer, dtype = param['dtype'])
        for j in tqdm(range(n_chunks), desc='Saving matrix', unit='chunk', leave=True):
            inputs = [input_[0][nf*j:nf*(j+1)] for input_ in tuples]+[Iapps[nf*j:nf*(j+1)]]
            z[nf*j:nf*(j+1)] = matriu_Fx(param, inputs)
    
    ############# DISTRIBUTED CLIENT CONFIGURATION #############
    
    worker_kwargs = {'processes': param['processes'],
                     'n_workers': param['n_workers'], #ha de ser un num. parell
                     'threads_per_worker': param['threads_per_worker'],
                     'silence_logs': 40, #para no mostrar info de warnings
                     'memory_limit': param['memory_limit'], #per worker
                     'memory_target_fraction': 0.95,
                     'memory_spill_fraction': 0.99,
                     'memory_pause_fraction': False,
                     'local_dir': param['client_temp_data']
                     }

    # do not kill worker at 95% memory level - es reinicia.
    dask.config.set({"distributed.worker.memory.terminate": False})

    # setup Dask distributed client
    cluster = LocalCluster(**worker_kwargs)
    client = Client(cluster)
    print(client)    
    
    ## Cálculos con dask.distributed
    f = zarr.open(param['matrix_folder']+'/matriu.zarr', 'r')
    FX = da.from_array(f, chunks=(files_chunk, wavelons))
    print('Saving covariance matrix FTF')
    if not param['recovery_FTF']:
        FTF = dd.from_dask_array(da.dot(FX.T, FX), columns = [str(elem) for elem in np.arange(wavelons)])
        FTF = FTF.persist() #carga tareas en segundo plano, cálculos pesado mejora el rendimiento.
        progress(FTF) #para ver la barra de progreso en distributed
        FTF.to_parquet(param['matrices_folder']+'/FTF.parquet')
    for i in range(param['recovery_var'],len(tuples)):
        client.restart() #para borrar tareas que se hayas quedado en caché.
        print('\n'+'--- Aproximando', var[i], '---')
        target = tuples[i][1].reshape(len(tuples[i][1]), 1)
        out_of_core_training(param, FX, target, files_chunk, var[i])
    print('\n')
                                
### LOSS
def out_of_core_training(param, FX, target, files_chunk, var):
    FTF = np.array(pd.read_parquet(param['matrices_folder']+'/FTF.parquet'))
    vaps = eigh(FTF, eigvals_only=True)
    #per matriu més grans de 2¹⁵ amb eigh() de numpy peta. pero amb scipy no
    A = FTF + np.identity(FTF.shape[1])*param['regularizer']*vaps[-1].real
    
    Y = da.from_array(target, chunks = (files_chunk, 1))
    FTY = dd.from_dask_array(da.dot(FX.T, Y), columns = ['0'])
    FTY = FTY.persist()
    progress(FTY)
    FTY.to_parquet(param['matrices_folder']+'/FTY.parquet')
    b = np.array(pd.read_parquet(param['matrices_folder']+'/FTY.parquet'))

    weights = lstsq(A, b, rcond=None)[0] #vector columna
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

    for j,I in enumerate(tqdm(Iapps, desc='Predicció', unit=' integracions', leave=True)):
        normalized = normalize(param, (d_var['i_'+str(i)] for i in range(len(tuples))), np.array([I]))
        Fx = matriu_Fx(param, normalized)
        for i in range(len(tuples)):
            d_var['i_'+str(i)] = (Fx.dot(d_var['weights_'+str(i)]))[0] #n+1
            d_var['predicted_'+str(i)][j] = d_var['i_'+str(i)] #guardo la prediccio
            
    for i in range(len(tuples)): #resultats
        print('->', var[i],'RMSE:', np.sum((d_var['t_'+str(i)][param['rmse_eval']:]-d_var['predicted_'+str(i)][param['rmse_eval']:])**2)/np.sum((np.mean(d_var['t_'+str(i)][param['rmse_eval']:])-d_var['t_'+str(i)][param['rmse_eval']:])**2)*100,'%')
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
def set_algorithm(param, wavelons, marker = 0):
    memoria = param['points']*param['n_Iapp']*wavelons*param[param['dtype']]/(2**30) 
    #SIEMPRE SE TRABAJARÁ CON FLOAT64 pero la opción float32 está sobretodo para cálculos out-of-core y reducir espacio de disco.
    if (memoria <= psutil.disk_usage('/')[2]/(1024**3)*ratio_comp[param['resolution']]/1.1) or param['recovery']:
    # se supone que si recovery = True ya hay una carpeta FX donde importar la matriz.
    #1.1 para dar un margen del 10% de espacio al disco para el directorio temporal de distributed
        n_chunks, files_chunk = parametrization(param, wavelons, marker)
    else:
        print('\n','- ERROR - No hay suficiente memoria para la configuración de la Wavenet.')
        print('Available disk space:', round(psutil.disk_usage('/')[2]/(2**30), 2), 'GB', 'Required disk space:', round(memoria/ratio_comp[param['resolution']]*1.1, 2), 'GB')
        exit()
    return n_chunks, files_chunk

# troba el núm files_chunk més gran (el mín n_chunks) entre les fites donades.
def chunk_size(param, wavelons, marker):
    matrix_MB = param['points']*param['n_Iapp']*wavelons*param[param['dtype']]/(2**20)
    # aquí se parametriz a la matriz para trabajar con los datos ya guardados, por lo tanto, estarán en el formato f4 o f8 escogido, por eso hay que poner param[param['dtype']]
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
        
def parametrization(param, wavelons, marker):
    #marker es un marcador para saber de qué cambio vienes, si de cambiar las Iapps o el n_puntos.
    n_chunks = chunk_size(param, wavelons, marker)
    #els chunks sense compactar tenen tantes files com files_chunk
    print('') # fake salt de linia perque només es mostri la config. que estic modificant 
    print(display_configuration(param, wavelons, n_chunks))
    print('Change n_Iapps [i], nº of points [p], fita_chunk_inf [inf] or fita_chunk_sup [sup]?')
    answer = input("(i/p/inf/sup): ")
    if answer == 'i':
        new_Iapps = input('Enter new n_Iapp value: ')
        param['n_Iapp'] = int(new_Iapps) #sobrescribo param
        print('\n'*100)
        return set_algorithm(param, wavelons, 0)
    elif answer == 'p':
        #si tienes que arreglar problemas de configuración sólo tocar: n_Iapps o n_puntos
        new_punts = input('Enter new n_points value: ')
        param['points'] = int(new_punts)
        print('\n'*100)
        return set_algorithm(param, wavelons, 1)
    elif answer == 'inf':
        new_fita_inf = input('Enter new fita_inf value [MB]: ')
        param['fita_chunk_inf'] = int(new_fita_inf)
        print('\n'*100)
        return set_algorithm(param, wavelons, marker)
    elif answer == 'sup':
        new_fita_sup = input('Enter new fita_sup value [MB]: ')
        param['fita_chunk_sup'] = int(new_fita_sup)
        print('\n'*100)
        return set_algorithm(param, wavelons, marker)
    print('')
    return n_chunks, param['n_Iapp']*param['points']//n_chunks

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
    div = [1]+divisors(descomp_factorial(n_chunks)) #comença pels divisors més petit
    # le añado la unidad como divisor, por si no hay otros
    for i, elem in enumerate(div):
        if files_chunk*elem*wavelons*8/(2**30) >= psutil.virtual_memory().free/(2**30):
            # se generan datos en float64 por eso el 8 (bytes)
            if i == 0: print('- ERROR - Chunks demasiado grandes'); exit()
            return div[i-1] #per quedar-me amb l'element anterior al if
    return elem #retorno el més gran perquè el chunk/elem em cap a la RAM

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
    n_chunks, files_chunk = set_algorithm(param, wavelons)
    
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
    #guardo los targets generados
    [save_data('prova/target_' + var[i] + '.parquet', da.from_array(target[i])) for i in range(len(target))]
    
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

def display_configuration(param, wavelons, n_chunks):
    nbytes = param['points']*param['n_Iapp']*wavelons*param[param['dtype']] #nbytes matrix
    config = [['n_chunks', n_chunks,
               'rows_chunk', param['n_Iapp']*param['points']//n_chunks],
              ['n_points', param['points'],
               'n_Iapps', param['n_Iapp']],
              ['fita_chunk_inf', param['fita_chunk_inf'],
               'fita_chunk_sup', param['fita_chunk_sup']],
              ['Memoria F(x)', str(round(nbytes/(2**30), 2)) + ' GB',
               'Memoria/chunk', str(round(nbytes/n_chunks/(2**20),2)) + ' MB']]
    
    taula = SingleTable(config, ' WAVENET WITH '+ str(wavelons) +' WAVELONS '+'--> OUT-OF-CORE ')
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
        d_var['pred_2'] = Iapps
        targets  = targets+(Iapps,)
        var = var+['Iapp']
        # Diagrama de fases 2D
        fig = plt.figure()
        plt.plot(targets[1], targets[0], label='Target', color='blue', linestyle='-', lw = 0.6)
        plt.plot(d_var['pred_1'], d_var['pred_0'], label='WNN', color='orange', linestyle='-', lw = 0.5)
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
    plt.plot(time, target, label='Target', color='blue', linestyle='-', lw = 0.6)
    plt.plot(time, predicted_data, label='WNN', color='orange', linestyle='-', lw = 0.6)
    plt.legend()

    plt.subplot(212)
    plt.title('Randomised forced term')
    plt.xlabel('Steps')
    plt.ylabel('Iapp')
    plt.plot(time, Iapps, color='blue', marker=',', linestyle='-')
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.10, right=0.95, hspace=0.5, wspace=0.35)
    
    plt.savefig('Outputs_'+var+'.png')
    plt.show()
