# Wavenet_research

La última versión de la Wavenet (V6) usa sólo un algoritmo out-of-core. Además hay que compilar el programa escrito en Cython con la comanda: $ python3 setup.py build_ext --inplace

Y ya se puede ejecutar cualquier modelo con la configuración definida en el archivo configuration.py

Hay que tener instalados los paquetes de python3 del archivo script_pau y script_pips de la carpeta /guías_&_scripts. Primero hay que ejecutar script_pau. Es una secuencia de órdenes para la shell, en Linux, así que se puede ejecutar con $sudo ./script_pau en el directorio global o para un usuario concreto, pero fuera de un entorno virtual. Luego se ejecuta $./script_pips, el cual se puede ejecutar en un entorno virtual si se usa uno.
