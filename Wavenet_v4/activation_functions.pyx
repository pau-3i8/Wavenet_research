#cython: language_level = 3
#cython: boundscheck = False
#cython: wraparound = False
#cython: nonecheck = False
#cython: cdivision = False
#cython: profile = True
#cython: initializedcheck = False

"""
cada scheduling option/ directiva de compilación se explica en: http://docs.cython.org/en/latest/src/userguide/source_files_and_compilation.html#compiler-directives

no poso cdivision = true perque em dona problemes de calcul amb les divisions

para compilar: $ python3 setup.py build_ext --inplace
para comprovar qué pasa por el compilador de python y qué no: $ cython -a arxiu.pyx
"""

from cython.parallel import parallel, prange
cimport numpy as np
import numpy as np
cimport cython

#################################### FUNCIONES DE ACTIVACIÓN ####################################

#El GIL tienen que estar activo para todas las interacciones entre funciones python, pero si se pueden escribir funciones en puro C (cdef functions) se puede liberar. Sin el GIL sólo se pueden llamar funciones que no usen gil. Sin este intérprete activo el código córre más rápido.
@cython.profile(False)
cdef inline double haar(double x) nogil:
    if 0<= x <= 1: return 1
    return 0

@cython.profile(False)
cdef inline double hat(double x) nogil: 
    if 0 <= x <= 2:
        if 0 <= x <= 1: return x
        elif 1 < x <= 2: return -x+2
    return 0
    
#si retorno els vector iterant a cada element a la funcio quadratic, i faig profile=false per no perdre temps cridant-la (hi ha overhead), faig el codi més ràpid perquè és una fucnió petita que la crido un colló de cops.
@cython.profile(False)
cdef inline double quadratic(double x) nogil:
    if 0 <= x <= 3:
        #és més ràpid multiplicar x*x mateix que elevar-ho a la 2
        if 0<= x <= 1: return 1/2*x*x 
        elif 1 < x <= 2: return -x*x+3*x-3/2
        elif 2 < x <= 3: return 1/2*x*x-3*x+9/2
    return 0

@cython.profile(False)
cdef inline double bicubic(double x) nogil:
    if 0 <= x <= 4:
        if 0 <= x <= 1: return 1/6*x*x*x
        elif 1 < x <= 2: return -1/2*x*x*x+2*x*x-2*x+2/3
        elif 2 < x <= 3: return 1/2*x*x*x-4*x*x+10*x-22/3
        elif 3 < x <= 4: return -1/6*x*x*x+2*x*x-8*x+32/3
    return 0

# select_phi_scaled
#si parallelitzo amb prange i tinc mols elements com quan construeixo la matriu, la cosa va ràpida, si parallelitzo a la simulació que vaig integració per integració, va molt lent.
@cython.profile(False)
cdef inline double phi(unicode name, double x, size_t n, size_t n_sf) nogil:
    # la función tiene que explorar todos los if's statements antes de encontrar la función que se usa referenciado con el tring. Si todo fuera if's tendria que explorar-lo sigual, así que mejor pongo 'quadratic' como el primer statement que es el que voy a usar y que tarde más si se usa otra función que no 'acostumbre' a usar.
    if name == 'quadratic':
        return quadratic(sup(x, n, 3, n_sf))
    elif name == 'haar':
        return haar(sup(x, n, 1, n_sf))
    elif name == 'hat':
        return hat(sup(x, n, 2, n_sf))
    elif name == 'bicubic':
        return bicubic(sup(x, n, 4, n_sf))
     
# select_psi
@cython.profile(False)
cdef inline double psi(unicode name, double x, size_t n, size_t n_sf) nogil:
    #idem que en la fucnión phi para el orden de los if's
    if name == 'quadratic':
        return 1/4*quadratic(2*sup(x, n, 3, n_sf)) - 3/4*quadratic(2*sup(x, n, 3, n_sf)-1) + 3/4*quadratic(2*sup(x, n, 3, n_sf)-2) - 1/4*quadratic(2*sup(x, n, 3, n_sf)-3)
    elif name == 'haar':
        return haar(2*sup(x, n, 1, n_sf))-haar(2*sup(x, n, 1, n_sf)-1)
    elif name == 'hat':
        return 1/2*hat(2*sup(x, n, 2, n_sf))-hat(2*sup(x, n, 2, n_sf)-1)+1/2*hat(2*sup(x, n, 2, n_sf)-2)
    elif name == 'bicubic':
        return 1/8*bicubic(2*sup(x, n, 4, n_sf))-1/2*bicubic(2*sup(x, n, 4, n_sf)-1)+3/4*bicubic(2*sup(x, n, 4, n_sf)-2)-1/2*bicubic(2*sup(x, n, 4, n_sf)-3)+1/8*bicubic(2*sup(x, n, 4, n_sf)-4)
    
@cython.profile(False)
cdef inline double sup(double x, size_t n, size_t d, size_t n_sf) nogil:
    if n_sf <= 2: return (x-1/2 + (n+1)/(n_sf+1))*d
    if n_sf > 2: return (x-1/2 + n/(n_sf-1))*d
    
########################### MATRIZ FUNCIONES DE ACTIVACIÓN 3 ENTRADAS ###########################

#Cython no puede definir algunos tipos de varibles si pueden ser de distintos tipos, como variable que puedan ser del tipo float o double. Para ello se puede usar las fused types.

# Para arrays np.ndarray[dtype_t, ndim=1, mode = 'c'] se ha substituido por dtype_t[::1]. Mejor usar tipos de variable _t para trabajar con C. En este caso se ha definido la contiguidad de los datos en fila como en C, pero se puede definir en columna, como en Fortran si se desea, para hace operaciones matriciales, por ejemplo, y combinar los dos contiguedades para multiplicar filas por columnas a mayor velocidad.

# se podría usar:
# with no gil, parallel():
#     for i in prange(): #pero no ha hecho falta.

#no puedes crear diccionarios es cython, una opción seria crear estructuras de datos pero es un lio.
def matriu_2D(object param, double[::1] input_1 not None, double[::1] input_2 not None,
              double[::1] Iapp not None, size_t wavelons):
              
    cdef str sf_name = param['fscale']
    cdef size_t n_sf = param['n_sf']
    cdef long int res = param['resolution']
    cdef size_t N = input_1.shape[0], i = 0, j, n1, n2, n3
    cdef long int c1, c2, c3, m
    
    cdef double[:,::1] matriu = np.zeros((wavelons, N), dtype = param['dtype'])
    
    ## Creas las columnas de la parte lineal
    if param['bool_lineal']:
        i = 2
        matriu[0,:] = input_1
        matriu[1,:] = input_2
    
    ## Creas las columnas de funciones de escala
    if param['bool_scale']:
        for n1 in range(n_sf):
            for n2 in range(n_sf):
                for n3 in range(n_sf):
                    for j in range(N):
                        matriu[i][j] = phi(sf_name, input_1[j], n1, n_sf)* phi(sf_name, input_2[j], n2, n_sf)*phi(sf_name, Iapp[j], n3, n_sf)
                    i+=1

    ## Creas las columnas de wavelets
    for m in range(res+1):
        #Les K's corresponen als factors de l'equacio 6.11 del TFG amb els valors de la taula 6.2
        for n1 in range(n_sf):
            for n2 in range(n_sf):
                for n3 in range(n_sf):
                    for c1 in range(2**m):
                        for j in range(N):
                            matriu[i][j] = phi(sf_name, input_1[j], n1, n_sf)* phi(sf_name, input_2[j], n2, n_sf)* psi(sf_name, (2**m)* Iapp[j] - c1, n3, n_sf)
                        i+=1
                    for c1 in range(2**m):
                        for j in range(N):
                            matriu[i][j] = phi(sf_name, Iapp[j], n1, n_sf)* phi(sf_name, input_1[j], n2, n_sf)* psi(sf_name, (2**m)* input_2[j] - c1, n3, n_sf)
                        i+=1
                    for c1 in range(2**m):
                        for j in range(N):
                            matriu[i][j] = phi(sf_name, input_2[j], n1, n_sf)* phi(sf_name, Iapp[j], n2, n_sf)* psi(sf_name, (2**m)* input_1[j] - c1, n3, n_sf)
                        i+=1
                    for c1 in range(2**m):
                        for c2 in range(2**m):
                            for j in range(N):
                                matriu[i][j] = phi(sf_name, input_1[j], n1, n_sf)* psi(sf_name, (2**m)* input_2[j] - c1, n2, n_sf)* psi(sf_name, (2**m)* Iapp[j] - c2, n3, n_sf)
                            i+=1
                    for c1 in range(2**m):
                        for c2 in range(2**m):
                            for j in range(N):
                                matriu[i][j] = phi(sf_name, Iapp[j], n1, n_sf)* psi(sf_name, (2**m)* input_1[j] - c1, n2, n_sf)* psi(sf_name, (2**m)* input_2[j] - c2, n3, n_sf)
                            i+=1
                    for c1 in range(2**m):
                        for c2 in range(2**m):
                            for j in range(N):
                                matriu[i][j] = phi(sf_name, input_2[j], n1, n_sf)* psi(sf_name, (2**m)* Iapp[j] - c1, n2, n_sf)* psi(sf_name, (2**m)* input_1[j] - c2, n3, n_sf)
                            i+=1
                    for c1 in range(2**m):
                        for c2 in range(2**m):
                            for c3 in range(2**m):
                                for j in range(N):
                                    matriu[i][j] = psi(sf_name, (2**m)* input_1[j] - c1, n1, n_sf)* psi(sf_name, (2**m)* input_2[j] - c2, n2, n_sf)* psi(sf_name, (2**m)* Iapp[j] - c3, n3, n_sf)
                                i+=1
    return np.asarray(matriu).T

########################### MATRIZ FUNCIONES DE ACTIVACIÓN 4 ENTRADAS ###########################
    
def matriu_3D(object param, double[::1] input_1 not None, double[::1] input_2 not None,
              double[::1] input_3 not None, double[::1] Iapp not None, size_t wavelons):
              
    cdef str sf_name = param['fscale']
    cdef size_t n_sf = param['n_sf']
    cdef long int res = param['resolution']
    cdef size_t N = input_1.shape[0], i = 0, j, n1, n2, n3, n4
    cdef long int c1, c2, c3, c4, m
    
    cdef double[:,::1] matriu = np.zeros((wavelons, N), dtype = np.double)
    
    if param['bool_lineal']:
        i = 3
        matriu[0,:] = input_1
        matriu[1,:] = input_2
        matriu[2,:] = input_3
    
## Creas las columnas de funciones de escala
    if param['bool_scale']:
        for n1 in range(n_sf):
            for n2 in range(n_sf):
                for n3 in range(n_sf):
                    for n4 in range(n_sf):
                        for j in range(N):
                            matriu[i, j] = phi(sf_name, input_1[j], n1, n_sf)* phi(sf_name, input_2[j], n2, n_sf)* phi(sf_name, input_3[j], n3, n_sf)* phi(sf_name, Iapp[j], n4, n_sf)
                        i+=1

    ## Creas las columnas de wavelets
    for m in range(res+1):
        #Les K's corresponen als factors de l'equacio 6.11 del TFG amb els valors de la taula 6.2input_3
        for n1 in range(n_sf):
            for n2 in range(n_sf):
                for n3 in range(n_sf):
                    for n4 in range(n_sf):
                        for c1 in range(2**m):
                            for j in range(N):
                                matriu[i, j] = phi(sf_name, input_2[j], n1, n_sf)* phi(sf_name, input_1[j], n2, n_sf)* phi(sf_name, input_3[j], n3, n_sf)* psi(sf_name, (2**m)* Iapp[j] - c1, n4, n_sf)
                            i+=1
                        for c1 in range(2**m):
                            for j in range(N):
                                matriu[i, j] = phi(sf_name, Iapp[j], n1, n_sf)* phi(sf_name, input_2[j], n2, n_sf)* phi(sf_name, input_1[j], n3, n_sf)* psi(sf_name, (2**m)* input_3[j] - c1, n4, n_sf)
                            i+=1
                        for c1 in range(2**m):
                            for j in range(N):
                                matriu[i, j] = phi(sf_name, input_3[j], n1, n_sf)* phi(sf_name, Iapp[j], n2, n_sf)* phi(sf_name, input_2[j], n3, n_sf)* psi(sf_name, (2**m)* input_1[j] - c1, n4, n_sf)
                            i+=1
                        for c1 in range(2**m):
                            for j in range(N):
                                matriu[i, j] = phi(sf_name, input_1[j], n1, n_sf)* phi(sf_name, input_3[j], n2, n_sf)* phi(sf_name, Iapp[j], n3, n_sf)* psi(sf_name, (2**m)* input_2[j] - c1, n4, n_sf)
                            i+=1
                        for c1 in range(2**m):
                            for c2 in range(2**m):
                                for j in range(N):
                                    matriu[i, j] = phi(sf_name, input_2[j], n1, n_sf)* phi(sf_name, input_1[j], n2, n_sf)* psi(sf_name, (2**m)* input_3[j] - c1, n3, n_sf)* psi(sf_name, (2**m)* Iapp[j] - c2, n4, n_sf)
                                i+=1
                        for c1 in range(2**m):
                            for c2 in range(2**m):
                                for j in range(N):
                                    matriu[i, j] = phi(sf_name, Iapp[j], n1, n_sf)* phi(sf_name, input_2[j], n2, n_sf)* psi(sf_name, (2**m)* input_1[j] - c1, n3, n_sf)* psi(sf_name, (2**m)* input_3[j] - c2, n4, n_sf)
                                i+=1
                        for c1 in range(2**m):
                            for c2 in range(2**m):
                                for j in range(N):
                                    matriu[i, j] = phi(sf_name, input_3[j], n1, n_sf)* phi(sf_name, Iapp[j], n2, n_sf)* psi(sf_name, (2**m)* input_2[j] - c1, n3, n_sf)* psi(sf_name, (2**m)* input_1[j] - c2, n4, n_sf)
                                i+=1
                        for c1 in range(2**m):
                            for c2 in range(2**m):
                                for j in range(N):
                                    matriu[i, j] = phi(sf_name, input_1[j], n1, n_sf)* phi(sf_name, input_3[j], n2, n_sf)* psi(sf_name, (2**m)* Iapp[j] - c1, n3, n_sf)* psi(sf_name, (2**m)* input_2[j] - c2, n4, n_sf)
                                i+=1
                        for c1 in range(2**m):
                            for c2 in range(2**m):
                                for j in range(N):
                                    matriu[i, j] = phi(sf_name, input_3[j], n1, n_sf)* phi(sf_name, input_2[j], n2, n_sf)* psi(sf_name, (2**m)* input_1[j] - c1, n3, n_sf)* psi(sf_name, (2**m)* Iapp[j] - c2, n4, n_sf)
                                i+=1
                        for c1 in range(2**m):
                            for c2 in range(2**m):
                                for j in range(N):
                                    matriu[i, j] = phi(sf_name, input_1[j], n1, n_sf)* phi(sf_name, Iapp[j], n2, n_sf)* psi(sf_name, (2**m)* input_3[j] - c1, n3, n_sf)* psi(sf_name, (2**m)* input_2[j] - c2, n4, n_sf)
                                i+=1
                        for c1 in range(2**m):
                            for c2 in range(2**m):
                                for c3 in range(2**m):
                                    for j in range(N):
                                        matriu[i, j] = psi(sf_name, (2**m)* input_2[j] - c1, n1, n_sf)* psi(sf_name, (2**m)* input_1[j] - c2, n2, n_sf)* psi(sf_name, (2**m)* input_3[j] - c3, n3, n_sf)* phi(sf_name, Iapp[j], n4, n_sf)
                                    i+=1
                        for c1 in range(2**m):
                            for c2 in range(2**m):
                                for c3 in range(2**m):
                                    for j in range(N):
                                        matriu[i, j] = psi(sf_name, (2**m)* Iapp[j] - c1, n1, n_sf)* psi(sf_name, (2**m)* input_2[j] - c2, n2, n_sf)* psi(sf_name, (2**m)* input_1[j] - c3, n3, n_sf)* phi(sf_name, input_3[j], n4, n_sf)
                                    i+=1
                        for c1 in range(2**m):
                            for c2 in range(2**m):
                                for c3 in range(2**m):
                                    for j in range(N):
                                        matriu[i, j] = psi(sf_name, (2**m)* input_3[j] - c1, n1, n_sf)* psi(sf_name, (2**m)* Iapp[j] - c2, n2, n_sf)* psi(sf_name, (2**m)* input_2[j] - c3, n3, n_sf)* phi(sf_name, input_1[j], n4, n_sf)
                                    i+=1
                        for c1 in range(2**m):
                            for c2 in range(2**m):
                                for c3 in range(2**m):
                                    for j in range(N):
                                        matriu[i, j] = psi(sf_name, (2**m)* input_1[j] - c1, n1, n_sf)* psi(sf_name, (2**m)* input_3[j] - c2, n2, n_sf)* psi(sf_name, (2**m)* Iapp[j] - c3, n3, n_sf)* phi(sf_name, input_2[j], n4, n_sf)
                                    i+=1
                        for c1 in range(2**m):
                            for c2 in range(2**m):
                                for c3 in range(2**m):
                                    for c4 in range(2**m):
                                        for j in range(N):
                                            matriu[i, j] = psi(sf_name, (2**m)* input_2[j] - c1, n1, n_sf)* psi(sf_name, (2**m)* input_1[j] - c2, n2, n_sf)* psi(sf_name, (2**m)* input_3[j] - c3, n3, n_sf)* psi(sf_name, (2**m)* Iapp[j] - c4, n4, n_sf)
                                        i+=1
    
    return np.asarray(matriu).T
    
# Possibles mejoras:
# 1- no tener que retornar nada y modificar un input en las funciones de activación
# 2- en vez de trabajar con typed memoryview para la matriz trabajar con malloc memoryview o mejor: C pointers (que seria lo millor)
