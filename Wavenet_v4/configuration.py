from numba.typed import Dict
from numba import types
import os, dask
import Wavenet

################################### DISTRIBUTED ENV. VARIABLES ###################################

# Variables de entorno de dask.distributed
os.environ['DASK_DISTRIBUTED__WORKER__MEMORY__TARGET'] = 'False'
os.environ['DASK_DISTRIBUTED__WORKER__MEMORY__SPILL'] = 'False'
os.environ['DASK_DISTRIBUTED__WORKER__MEMORY__PAUSE'] = '0.93'
os.environ['DASK_DISTRIBUTED__WORKER__MEMORY__TERMINATE'] = '0.95'
os.environ['DASK_DISTRIBUTED__SCHEDULER__ALLOWED_FAILURES'] = '10'
os.environ['DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT'] = '5'
os.environ['DASK_DISTRIBUTED__COMM__RETRY__COUNT'] = '10'

# Variable de entorno para numpy
#os.environ['OMP_NUM_THREADS']='1' #limita el número de threads de numpy

###################################### WAVENET CONFIGURATION ######################################

#nivell 3 (1) para 3D es el máximo para la workstation con n_sf = 5 o nivel 4 (2) con n_sf = 3
#para aumentar de nivel se puede calcular directamente los pesos sin guardar FTF para el cálculo de valores propios para el uso del regularizador. Pero se tendría que sumar un término mínimo a la diagonal de Fx, para evitar errores con la inversa al calcular los pesos. Eso implicaría forzar chunks quadrados como el primer algoritmo que parametrizaba la red y limita mucho la libertad de config.

param0 = {'dtype':'float64',
          'float64':8,
          'float32':4,
          
          'domain_margin':2.5,
          'sec_factor':1.5,
          'regularizer':1e-30,
          
          'points':5000,
          'n_Iapp':2900,
          
          'points_simu':5000,
          'n_Iapp_simu':50,
          'rmse_eval':0,
          
          'I_max':0.1,
          'I_min':0,
          
          'resolution':0,
          'n_sf':4,
          'fscale':'quadratic',
          
          'bool_lineal':True,
          'bool_scale':True,
          
          'generateIapp':True,
          'generateIapp_simu':True,
          'shuffle':False,
          
          'fita_chunk_inf':130,
          'fita_chunk_sup':140,
          
          'compression':6,
               
          'recovery':False,
          'recovery_FTF':False,
          'recovery_var':0,
          
          'matrix_folder':'temporary_dir/FX',
          'matrices_folder':'temporary_dir',
          'client_temp_data':'temporary_dir',
          'results_folder':'resultados',
          
          'only_simulation':False,
}

# Guía de parámetros.

"""
- dtype: sisrve para definir el tipo de variable. Puede ser float64 o float32 (se importan desde numba, por eso no son strings como con numpy). Float32 ocupa la mitad de espacio, és menos preciso pero para algunas simulaciones puede bastar y se consigue manejar el doble de datos.
- float64: define el número de bytes que ocupa el tipo de variable float64. - No tocar -
- float32: define el número de bytes que ocupa el tipo de variable float32. - No tocar -
- domain_margin: Toma valor 2.5 definido en un tanto por 1 del aumento del dominio, respecto a la I_max, para dar margen al dominio de Iapps en el que se aproxima la red para poder obtener extremos más mayores en el dominio de cada variable y hacer una normalización sin problemas. Si el espacio vectorial da problemas a 2.5 del dominio de Iapps se puede bajar un poco, pero para los modelos probados no han habido problemas hasta ahora.
- sec_factor: Es un factor de seguridad, necesario debido al comportamiento del software, que evita problemas al asignar el tamaño de los archivos, en los que se divide la matriz FX, a cada hilo del procesador para genenrarla en paralelo. Ya que cada archivo se tiene que cargar primero a la memoria para poder guardarlo luego. Toma valores de tipo float. Si se cambia el tamaño del chunk con las fitas, es probable que haya que modificar el factor de seguridad. A mayor chunk, mayor el factor de seguridad. Lo ideal seria controlar los recursos mientras se genera la primer atanda de archivos para comprobar que no se está ahogando la RAM. Se puede controlar en la shell con el comando: 'htop -d 0.9' (-d 0.9 sirve para refrescar los datos cada 0.9 segundos, para ver los cambios con mayor fluidez).
- regularizer: Define el parámetro de regularización para evitar problemas de inversa y suavizar la superficie de curvatura del modelo. A 1e-30 le añade un valor derivado de la imprecision computacional suficiente para el uso que tiene. Para saber hasta que decimal de precision se puede llegar con cada tipo de variable se puede ejecutar en python3: >>> numpy.finfo(tipo_de_la_variable).eps .Si se trabaja con 'float32' la máxima precisión és de 1.1920929e-07 y 1e-30 sólo añade el mismo número que qualquier valor por debajo de 1.1920929e-07.
- points: (valor enteror) Corresponde al número de integraciones por Iapp en el euler para la approximación. Sugiero utilizar el mínimo necesario para que la dinámica dle sitema llegue a los puntos fijos y poder exprimir con el  número de Iapps. También hay que tener en cuenta el paso de integración por eso, definido en euler_dict0['h']. Si se quiere usar un número muy grande de integraciones des de la configuración, és possible que la mágina se quede sin memoria si se mate la ejecución del programa, porque primero se calculan los límites de los dominios de cada variable para normalizar en el post procesado y en ese cálculo se cogen en numero de puntos definidos en la configuración junto con 10.000 Iapps randomizadas. Recomiendo empezar con un numero de puntos suficiente para definir los puntos fijos del sistema y luego cambiar el numero de puntos en la terminal durante la ejecución.
- n_Iapp: (valor enteror) Corresponde al número de Iapps usadas para la aproximación.
- points_simu: Corresponde al número de integraciones de la simulación. La simulación no és tan rapida porque tiene que hacer una muntiplicación de vectores a cada integración para poder el punto de la siguiente integración, si se han usado muchos wavelons puede ser lento. A menos que sea para poder obtener resultados definitivos recomiendo usar pocos puntos.
- n_Iapps_simu: Lo mismo que con la sintegraciones, mientra se vea que aproxima bien qualquier Iapp aleatoria, sean 1, 2, 2000 da igual. Poner valores pequeños mientras se hagan pruebas y ya se podndrá un valor grande para gráficos deficnitivos y demostrar que la red puede aproximar qualquier Iapp aleatoria dentro del dominio.
- rmse_eval: Corresponde al número de integraciones a partir del cuál se empieza a calcular el error relativo. Pensado para poder eliminar el transitorio y calcular el error de las dos funciones una vez sincronizadas. La idea es que el valor de este parámetro sea alrededor de un 8% del total de integraciones, para poder calcular el error a partir de allí. No escatimar en el número de integraciones para el análisis, ya que la simulacion és muy rápida.
- I_max: Sirve para definit la I máxima del dominio de Iapps a estudiar.
- I_min: Sirve para definit la I mínima del dominio de Iapps a estudiar. Una possible sugerencia para definir un dominio en cada modelo es:
		 FHN    -> 'I_max':  0.1, 'I_min': 0.0
		 Wang   -> 'I_max':  3.0, 'I_min':-3.0
		 MLecar -> 'I_max':100.0, 'I_min':10.0
- resolution: Corresponde al valor asociado a los niveles de resolución de la red. Toma valores enteros, inferior a 0 si no se quieren usar wavelets. Valor 0 para el primer nivel de wavelets, 1 para el segundo nivel y así sucesivamente.
- n_sf: corresponde al número de funciones superpuestas. Toma valores 1, 3, 5 y 7. 5 es el que da mejores resultados sin penalizar demasiado el tiempo de cálculo.
- fscale: Es un string que define el tipo de funcion de escala a usar y a su vez la función de escala con la que se construyen las wavelets. Pueden ser: 'haar' (no recomendado excepto para pruebas teóricas), 'hat', 'quadratic', 'bicubic'.
- bool_lineal: Es un parámetro booleano para definir si se desea utilizar coeficietes lineales (True) o no (False) en la red. (recomiendo usar True, no penaliza nada y mejora las aproximaciones).
- bool_scale: Es un parámetro booleano para definir si se desea utilizar el primer nivel de sólo funciones de escala (True) o no (False) en la red.
- generateIapp: Es un parámetro booleano para definir si se desea randomizar las Iapps para la aproximación (True) o no (False). Lógicamente casi siempre estará en True, pero si se desea hacer un benchmark entre dos algoritmos parecidos para ver qué tal aproximan los mismos datos, se puede definir con un False y entonces la red importará los Iapps desde un archivo .parquet. La red randomiza datos a partir de unas semillas (seeds) a lo largo del código, así que se generaran las mismas Iapps con una misma configuración de la red si el parámetro está en True.
- generate_Iapp_simu: Es un parámetro booleano para definir si se desea randomizar las Iapps para la simulación (True) o no (False). Para hacer tests rápidos puede ser bueno definirlo False y simplemente mirar el error, pero lógicamente para resultados definitivos tendría que ser True.
- shufle: Es un parámetro booleano para definir si se desea mezclar las Iapps importadas en la simulación (generate_Iapp_simu = False) o no (False). Puede ser interesante comprobar la robusted aunque se hagan tests rápidos. Los 'tests rápidos' de los que he hablado són prácitcos para simular una red con una configuración y reutilizar las mismas Iapps de la aproximación para la simulación y visto el error de predicción modificar un parámetro como el regularizados o el número de funciones superpuestas y recalcular el error con los mismos números de Iapps y comparar los errores de las dos simulaciones para encontrar la mejor puesta en marxa. Si entre estas comparaciones el número de Iapps cambia, se randomizará otra tanda de Iapps a pesar de las semillas usadas en el código, y dejaran de ser comparativas válidas.
- fita_chunk_inf: Define los MB mínimos del chuhnk de la matriz FX. - No tocar si no se sabe lo que se hace -
- fita_chunk_sup: Define los MB máximos del chuhnk de la matriz FX. - No tocar si no se sabe lo que se hace -
Con las dos fitas inferiores se puede definir un rango aceptable de dimensiones de chunks que pueden ser buenos para nuestra máquina en particular. EL tamaño de los chunks es muy important para que puedan realizarse los cálculos en Dask sin problemas de moemoria. El tamanyo óptimo de los chunks depende del número tareas a ejecutar, la cantidad de memoria RAM, la cantidad de hilos del procesador usados en el cliente de dask.distributed y el tipo de cálculos. Los valores predefinidos son suficientes.
- compression: Define el nivel (índice) de compresion del algoritmo zstd de los archivos hdf5 para la generación de la matriz de funciones de activación Fx. Los valores de compresión van del 1 al 9 en valores enteros. Al generar la matriz en paralelo poner el índice a 6 no penaliza demasiado y tiene un muy buen ratio de compresión. El índice 1 o 2 puede ser una buena elección para discos duros lentos. Se ha mejorado mucho respecto a la anterior compresión usada. La utilización de la RAM con este algoritmo de ocmpresió és muy reducida, así que se puede empezar a definir el sec_factor a 1 y subir poco a poco si hace falta a 1.1, 1.15...
- recovery: A veces pueden haber problemas a lo largo de la simulación, si se hace out-of-core pueden tardar muchas horas si se es ambicioseo. A lo largo de estas horas uno no sabe l oque puede pasar, personalmente empujo el ordenador un poco al límite para sacarle potencia y eso genera inestabilidades, he tenido que reiniciar el algoritmo multiples veces luego de apagones repentinos. Los parámetros recovery evitaran que estas inestabilidades sean un menor dolor de cabeza. Si se ha guardado la matriz FX, no la volvera a generar con True y al ejecutar la red directamente importará la matriz para empezar a calcular la de covariancia FTF. Si no hay nada que recuperar y se ejecuta la red con normalidad tiene que ser False.
- recovery_FTF: Igual que con el caso anterior, si se ha llegado a guardar la a matriz FTF, no la volvera a generar con True y al ejecutar la red directamente importará la matriz para empezar a calcular los pesos. Si no hay nada que recuperar y se ejecuta la red con normalidad tiene que ser False.
- recovery_var: Si se estan ejecutando los pesos y se llega a guardar alguno de ellos este parámetro permite empezar a calcular el peso siguiente donde se había dejado todo. El parámetro toma valores enteros del 0 al número de variables de estado del modelo. 0 para calcular todos los pesos  apartir del primero (como si fuera una ejecución con normalidad), 1 para calcular todos los pesos a partir de la segunda variable de estado, 2 para calcular todos los pesos a partir de la tercera variable de estado, y así sucesivamente.
- matrices_folder: Define el directorio (en formato string) donde se guardan las matrices FTF y el resto de matrices temporales. Se creará un directorio en la carpeta en la que se ejecute el código de la red.
- matrix_folder: Define el directorio (en formato string) donde se guarda la matriz FX Se creará un directorio en la carpeta en la que se ejecute el código de la red. OBLIGATORIO que sea un subdirectorio del definido en 'matrices_folder' tal y como esta puesto en default.
- client_temp_data: Define el directorio (en formato string) donde se guardan los datos temporales del cliente del cluster de distributed.
- results_folder: Define el directorio (en formato string) donde se guardan los vectores de los pesos y los outputs d ela predicción. Se creará un directorio en la carpeta en la que se ejecute el código de la red.
- only_simulation: Es un parámetro booleano para definir si se desea ejecutar sólo la simulación (True) o no (False) de la red. És útil para, una vez encontrados los pesos, estudiar rangos de Iapps más detalladamente con distintas simulaciones sin perder tiempo.
"""

## Setup files first, the distributed local_dir path is in a WN configured folder
Wavenet.set_files(param0)

################################## EULER'S PARAMETER DICCIONARY ##################################

# Dict with keys as string and values of type float
euler_dict0 = Dict.empty(key_type=types.unicode_type,
                         value_type=types.float64)
# paso de integración
euler_dict0['h'] = 0.1 # en Nagumo h = 0.1 | en Wang h = 0.005 | en Morris h = 0.5
# Parámetros Nagumo 2D
euler_dict0['a'] = 0.14
euler_dict0['gamma'] = 2.54
euler_dict0['eps'] = 0.1
# Parámetros Nagumo 3D (junto con los de Nagumo 2D)
euler_dict0['alpha'] = 0.02
euler_dict0['c'] = -0.775
euler_dict0['d'] = 1
euler_dict0['mu'] = 0.01
# Parámetros Wang
euler_dict0['gNa'] = 45.0
euler_dict0['vK'] = -80.0
euler_dict0['vNa'] = 55.0
euler_dict0['vL'] = -65.0
# Parámetros que comparten Wang y Morris-Lecar
euler_dict0['phi'] = 1/15 # en Wang phi = 4.0 | en Morris phi = 1/15
euler_dict0['gL'] = 2.0   # en Wang gL = 0.1  | en Morris gL = 2.0
euler_dict0['gK'] = 8.0   # en Wang gK = 18.0 | en Morris gK = 8.0
# Parámetros Morris-Lecar
euler_dict0['gCa'] = 4.0
euler_dict0['Cm'] = 20.0
euler_dict0['EL'] = -60
euler_dict0['EK'] = -84
euler_dict0['ECa'] = 120
euler_dict0['V1'] = -1.2
euler_dict0['V2'] = 18
euler_dict0['V3'] = 12
euler_dict0['V4'] = 17.4
# Parámetro de ruido
euler_dict0['noise'] = 0.08 #ruido del 8% (con una distribución uniforme varia de 0 a 0.08 el valor de la variable afectada -> con ruido)

##################################################################################################

def config():
    return param0, euler_dict0
