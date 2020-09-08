import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches #used to write custom legends

# You can use the following color code:
EQUILIBRIUM_COLOR = {'Stable node':'C0',
                    'Unstable node':'C1', 
                    'Saddle':'C4',
                    'Stable focus':'C3',
                    'Unstable focus':'C2',
                    'Center':'C5'}

def find_roots(a,alfa,beta,I):
    # The coeficients of the polynomial equation are:
    # 1             * v**3 
    # - (1+a)       * v**2 
    # (a-alfa/beta) * v**1 
    # - I           * v**0
    coef = [1, -1-a, a+alfa/beta, - I]
    roots = [np.real(r) for r in np.roots(coef) if np.isreal(r)]
    
    # We store the position of the equilibrium. 
    return [[r, r*(r-1)*(r-a) - I] for r in roots]

def jacobian_fitznagumo(v, w, a, alfa, beta, I):
    return np.array([[-3*v**2+2*v*(a+1)-a , 1],
                       [alfa, beta]])

def stability(jacobian):
    eigv = np.linalg.eigvals(jacobian)  
    if all(np.real(eigv)==0) and all(np.imag(eigv)!=0):
        nature = "Center" 
    elif np.real(eigv)[0]*np.real(eigv)[1]<0:
        nature = "Saddle"
    else: 
        nature = 'Unstable' if all(np.real(eigv)>0) else 'Stable'
        nature += ' focus' if all(np.imag(eigv)!=0) else ' node'
    return nature

"""
#alternativa per definir l'estabilitat, però el càlcul és més lent
def stability_alt(jacobian):
    determinant = np.linalg.det(jacobian)
    trace = np.matrix.trace(jacobian)
    if np.isclose(trace, 0):
        nature = "Center (Hopf)"
    elif np.isclose(determinant, 0):
        nature = "Transcritical (Saddle-Node)"
    elif determinant < 0:
        nature = "Saddle"
    else:
        nature = "Stable" if trace < 0 else "Unstable"
        nature += " focus" if (trace**2 - 4 * determinant) < 0 else " node"
    return nature
"""

### Plot the bifurcation diagram for v with respect to parameter I.
def solucio_punts(config):
    Iapps = np.linspace(0,0.3,40000) #calculo l'estabilitat en 40.000 Iapps
    I_list = []
    eqs_list = []
    nature_legends = []

    for I in Iapps:
        config['I'] = I
        roots = find_roots(**config)
        for v,w in roots:
            J = jacobian_fitznagumo(v,w, **config)
            nature = stability(J)
            nature_legends.append(nature)
            I_list.append(I)
            eqs_list.append(v)
    return nature_legends, I_list, eqs_list

## Diagrama de hopf
def hopf_diagram(param):
    config = {'a':param['a'], 'alfa':param['alfa'], 'beta':param['alfa']*param['gamma']}
    nature_legends, I_list, eqs_list = solucio_punts(config)
    fig, ax = plt.subplots(1,1,figsize=(10,5))
    labels = frozenset(nature_legends)
    ax.scatter(I_list, eqs_list, c=[EQUILIBRIUM_COLOR[n] for n in nature_legends], s=5.9)  
    ax.legend([mpatches.Patch(color=EQUILIBRIUM_COLOR[n]) for n in labels], labels, 
              loc='lower right')
    ax.set(xlabel='External stimulus, $I_{ext}$', 
           ylabel='Equilibrium Membrane potential, $v^*$');
    plt.grid(True)
    plt.show()

def limits(param):
    config = {'a':param['a'], 'alfa':param['alfa'], 'beta':param['alfa']*param['gamma']}
    nature_legends, I_list, _ = solucio_punts(config)
    #retorn la I_limit = unstable focus (inici i fi de I_oscilatoris)
    I_limit=[]
    for i in range(len(nature_legends)):
        if nature_legends[i] == 'Stable focus' and nature_legends[i+1] == 'Unstable focus':
            I_limit.append(I_list[i+1])
        elif nature_legends[i] == 'Unstable focus' and nature_legends[i+1] == 'Stable focus':
            I_limit.append(I_list[i])
        elif len(I_limit)==2:
            return I_limit
    #amb 200.000 Iapps calculats, I_limit=[0.036364681823409115, 0.14975924879624397]


##Execusió
#param = {'a':0.14, 'alfa':-0.01, 'gamma':2.54}
#hopf_diagram(param)

