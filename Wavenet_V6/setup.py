from setuptools import Extension, setup
from Cython.Build import cythonize

#a veces puede ser útil -ffast-math como extra_compile_arg para mejorar la velocidad.
# si se usaran funciones matmáticas comunes de la libreri math como cos() o exp() haría falta añadir libraries=["m"] en Extension(). -fopenmp se ha definido por si se hacía uso de procesamiento en parallelo.

ext_modules = [
    Extension(
        'activation_functions',
        ['activation_functions.pyx'],
        extra_compile_args=['-fopenmp', '-ffast-math'],
        extra_link_args=['-fopenmp'],
    )]

setup(ext_modules = cythonize(ext_modules, compiler_directives={'language_level':'3'}))
