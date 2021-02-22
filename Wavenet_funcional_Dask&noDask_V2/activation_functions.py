from numba import njit
import numpy as np

### PHIS
@njit(nogil = True, fastmath = True, cache = True)
def haar(inputs, dtype):
    x = np.zeros_like(inputs, dtype = dtype)
    for i, elem in enumerate(inputs):
        if 0<= elem <= 1:
            x[i] = 1
        else:
            x[i] = 0
    return x
    
@njit(nogil = True, fastmath = True, cache = True)
def hat(inputs, dtype):
    x = np.zeros_like(inputs, dtype = dtype)
    for i, elem in enumerate(inputs):
        if 0 <= elem <= 2:
            if 0 <= elem <= 1:
                x[i] = (elem)
            elif 1 < elem <= 2:
                x[i] = -(elem)+2
        else:
            x[i] = 0
    return x
    
@njit(nogil = True, fastmath = True, cache = True)
def quadratic(inputs, dtype):
    x = np.zeros_like(inputs, dtype = dtype)
    for i, elem in enumerate(inputs):
        if 0 <= elem <= 3:
            if 0<= elem <= 1:
                x[i] = (1/2*(elem)**2)
            elif 1 < elem <= 2:
                x[i] = (-(elem)**2+3*(elem)-3/2)
            elif 2 < elem <= 3:
                x[i] = (1/2*(elem)**2-3*(elem)+9/2)
        else:
            x[i] = 0
    return x
    
@njit(nogil = True, fastmath = True, cache = True)
def bicubic(inputs, dtype):
    x = np.zeros_like(inputs, dtype = dtype)
    for i, elem in enumerate(inputs):
        if 0 <= elem <= 4:
            if 0 <= elem <= 1:
                x[i] = (1/6*(elem)**3)
            elif 1 < elem <= 2:
                x[i] = (-1/2*(elem)**3+2*(elem)**2-2*(elem)+2/3)
            elif 2 < elem <= 3:
                x[i] = (1/2*(elem)**3-4*(elem)**2+10*(elem)-22/3)
            elif 3 < elem <= 4:
                x[i] = (-1/6*(elem)**3+2*(elem)**2-8*(elem)+32/3)
        else:
            x[i] = 0
    return x

# select_phi_scaled
def phi(name, inputs, n, dtype):
    if name == 'haar':
        x = superposition(inputs, n, 1) #escalado de 0 a 1
        return haar(x, dtype)
    if name == 'hat':
        x = superposition(inputs, n, 2) #escalado de 0 a 2
        return hat(x, dtype)
    if name == 'quadratic':
        x = superposition(inputs, n, 3) #escalado de 0 a 3
        return quadratic(x, dtype)
    if name == 'bicubic':
        x = superposition(inputs, n, 4) #escalado de 0 a 4
        return bicubic(x, dtype)

### PSIS
def haar_psi(inputs, dtype):
    return haar(2*inputs, dtype)-haar(2*inputs-1, dtype)

def hat_psi(inputs, dtype):
    return 1/2*hat(2*inputs, dtype)-hat(2*inputs-1, dtype)+1/2*hat(2*inputs-2, dtype)

def quadratic_psi(inputs, dtype):
    return 1/4*quadratic(2*inputs, dtype)-3/4*quadratic(2*inputs-1, dtype)+3/4*quadratic(2*inputs-2, dtype)-1/4*quadratic(2*inputs-3, dtype)

def bicubic_psi(inputs, dtype):
    return 1/8*bicubic(2*inputs, dtype)-1/2*bicubic(2*inputs-1, dtype)+3/4*bicubic(2*inputs-2, dtype)-1/2*bicubic(2*inputs-3, dtype)+1/8*bicubic(2*inputs-4, dtype)

# select_psi
def psi(name, inputs, n, dtype):
    if name == 'haar':
        x = superposition(inputs, n, 1) #escalado de 0 a 1
        return haar_psi(x, dtype)
    if name == 'hat':
        x = superposition(inputs, n, 2) #escalado de 0 a 2
        return hat_psi(x, dtype)
    if name == 'quadratic':
        x = superposition(inputs, n, 3) #escalado de 0 a 3
        return quadratic_psi(x, dtype)
    if name == 'bicubic':
        x = superposition(inputs, n, 4) #escalado de 0 a 4
        return bicubic_psi(x, dtype)

### DESPLAZAMIENTOS DE LA FUNCIÓN
def superposition(x, n, d):
    if n == 0:
        a = 0
    elif n%2 != 0:
        a =  d/(2**((n+1)/2))
    elif n%2 == 0:
        a =  -d/(2**(n/2))
    return x*d+a
