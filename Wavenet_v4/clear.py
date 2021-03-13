import sys, psutil, os, signal

######################################### WIPES RESOURCES #########################################

# vacía la memoria si se ha interrumpido la ejecucion de alguna simu anterior
def kill_processes(func):
    
    for process in psutil.process_iter():
        if process.cmdline() == ['python3', sys.argv[1]]:
            print('Process found. Terminating it.')
            os.kill(process.pid, signal.SIGKILL)
            break
        
if __name__ == "__main__":
    kill_processes(sys.argv[1])

#ejecutar con :$ python3 clear.py func.py
#donde func.py es Nagumo_3D.py o como se le llame al archivo con el euler.
