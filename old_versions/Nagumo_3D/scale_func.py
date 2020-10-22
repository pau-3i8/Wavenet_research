import numpy as np
from numba import njit

###PHIS
@njit(nogil = True, fastmath = True, parallel=False)
def haar(inputs):
    x = np.zeros_like(inputs)
    for i, elem in enumerate(inputs):
        if 0<= elem <= 1:
            x[i] = 1
        else:
            x[i] = 0
    return x

@njit(nogil = True, fastmath = True, parallel=False)
def hat(inputs):
    x = np.zeros_like(inputs)
    for i, elem in enumerate(inputs):
        if 0 <= elem <= 2:
            if 0 <= elem <= 1:
                x[i] = (elem)
            elif 1 < elem <= 2:
                x[i] = -(elem)+2
        else:
            x[i] = 0
    return x

@njit(nogil = True, fastmath = True, parallel=False)
def quadratic(inputs):
    x = np.zeros_like(inputs)
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

@njit(nogil = True, fastmath = True, parallel=False)
def bicubic(inputs):
    x = np.zeros_like(inputs)
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

def select_phi(name, inputs):
    if name == 'haar':
        return haar(inputs)
    if name == 'hat':
        return hat(inputs)
    if name == 'quadratic':
        return quadratic(inputs)
    if name == 'bicubic':
        return bicubic(inputs)

def select_phi_scaled(name, inputs, n):
    if name == 'haar':
        x = superposition(inputs, n, 1)
        return haar(x)
    if name == 'hat':
        x = superposition(inputs, n, 2) #escalat
        return hat(x)
    if name == 'quadratic':
        x = superposition(inputs, n, 3)
        return quadratic(x)
    if name == 'bicubic':
        x = superposition(inputs, n, 4)
        return bicubic(x)
        
###PSIS
def haar_psi(name, inputs):
    return select_phi(name, 2*inputs)-select_phi(name, 2*inputs-1)

def hat_psi(name, inputs):
    return 1/2*select_phi(name, 2*inputs)-select_phi(name, 2*inputs-1)+1/2*select_phi(name, 2*inputs-2)

def quadratic_psi(name, inputs):
    return 1/4*select_phi(name, 2*inputs)-3/4*select_phi(name, 2*inputs-1)+3/4*select_phi(name, 2*inputs-2)-1/4*select_phi(name, 2*inputs-3)

def bicubic_psi(name, inputs):
    return 1/8*select_phi(name, 2*inputs)-1/2*select_phi(name, 2*inputs-1)+3/4*select_phi(name, 2*inputs-2)-1/2*select_phi(name, 2*inputs-3)+1/8*select_phi(name, 2*inputs-4)

def select_psi(name, inputs, n):
    if name == 'haar':
        x = superposition(inputs, n, 1)
        return haar_psi(name, x)
    if name == 'hat':
        x = superposition(inputs, n, 2) #escalat
        return hat_psi(name, x)
    if name == 'quadratic':
        x = superposition(inputs, n, 3)
        return quadratic_psi(name, x)
    if name == 'bicubic':
        x = superposition(inputs, n, 4)
        return bicubic_psi(name, x)

###DESPLAÇAMENTS
def superposition(x, n, d):
    if n == 0:
        a = 0
    elif n%2 != 0:
        a =  d/(2**((n+1)/2))
    elif n%2 == 0:
        a =  -d/(2**(n/2))
    return x*d+a
