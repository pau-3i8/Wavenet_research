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
def phi(name, x, n, n_sf, dtype):
    if name == 'haar':
        inputs = superposition(x, n, 1, n_sf) #escalado de 0 a 1
        return haar(inputs, dtype)
    if name == 'hat':
        inputs = superposition(x, n, 2, n_sf) #escalado de 0 a 2
        return hat(inputs, dtype)
    if name == 'quadratic':
        inputs = superposition(x, n, 3, n_sf) #escalado de 0 a 3
        return quadratic(inputs, dtype)
    if name == 'bicubic':
        inputs = superposition(x, n, 4, n_sf) #escalado de 0 a 4
        return bicubic(inputs, dtype)

# select_psi
def psi(name, x, n, n_sf, dtype):
    if name == 'haar':
        inputs = superposition(x, n, 1, n_sf)
        return haar(2*inputs, dtype)-haar(2*inputs-1, dtype)
        
    if name == 'hat':
        inputs = superposition(x, n, 2, n_sf)
        return 1/2*hat(2*inputs, dtype)-hat(2*inputs-1, dtype)+1/2*hat(2*inputs-2, dtype)
        
    if name == 'quadratic':
        inputs = superposition(x, n, 3, n_sf)
        return 1/4*quadratic(2*inputs, dtype)-3/4*quadratic(2*inputs-1, dtype)+3/4*quadratic(2*inputs-2, dtype)-1/4*quadratic(2*inputs-3, dtype)
        
    if name == 'bicubic':
        inputs = superposition(x, n, 4, n_sf)
        return 1/8*bicubic(2*inputs, dtype)-1/2*bicubic(2*inputs-1, dtype)+3/4*bicubic(2*inputs-2, dtype)-1/2*bicubic(2*inputs-3, dtype)+1/8*bicubic(2*inputs-4, dtype)
        

def superposition(x, n, d, n_sf):
    #d és el reescalat
    #n_sf numero de funcions superposades
    #n index de la funció superposada. Si hi ha dos els indexos son 0 i 1 i la funcio 0 està a l'esquerra i la funcio 1 a la dreta.
    if n_sf <= 2: return ((x-1/2)+(n+1)/(n_sf+1))*d
    if n_sf > 2: return ((x-1/2)+n/(n_sf-1))*d
