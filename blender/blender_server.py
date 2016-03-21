# http://blender.stackexchange.com/questions/41533/how-to-remotely-run-a-python-script-in-an-existing-blender-instance

# Script to run from blender:
#   blender --python blender_server.py

import sys

global param_dict
# HOST = "127.0.0.1"
# HOST = "192.168.1.100"
PATH_MAX = 4096

def execfile(filepath):
    import os
    global_namespace = {
        "__file__": filepath,
        "__name__": "__main__",
        }
    with open(filepath, 'rb') as file:
        exec(compile(file.read(), filepath, 'exec'), global_namespace)

class FlushFile(object):
    '''
    http://stackoverflow.com/questions/230751/how-to-flush-output-of-python-print
    to flush after print
    '''

    def __init__(self, fd):
        self.fd = fd

    def write(self, x):
        ret = self.fd.write(x)
        self.fd.flush()
        return ret

    def writelines(self, lines):
        ret = self.writelines(lines)
        self.fd.flush()
        return ret

    def flush(self):
        return self.fd.flush

    def close(self):
        return self.fd.close()

    def fileno(self):
        return self.fd.fileno()


def main(PORT, HOST):
    import socket

    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversocket.bind((HOST, PORT))
    serversocket.listen(1)

    print("Listening on %s:%s" % (HOST, PORT))

    looping = True
    while looping:
        connection, address = serversocket.accept()
        buf = connection.recv(PATH_MAX)

        for filepath in buf.split(b'\x00'):
            if filepath:
                if filepath == b'exit()':
                    print('blender_server got exit() command thus shutting down blender program.')
                    looping = False
                    break
                else:
                    print("Executing:", filepath)
                    # sys.stderr.write(str(''.join(["Executing:", filepath])))
                    try:
                        execfile(filepath)
                    except:
                        import traceback
                        traceback.print_exc()

    serversocket.close()

if __name__ == "__main__":
    """
    This script is intended to be run in blender python interpreter.
     Thus it enables to remotely (by socket) execute another python scripts in blender.
     Advantage is the blender is always running in background so it has not to be started every time.
     The main functions loops forever waiting for socket input until the input is 'exit()'.

    """

    sys.stdout = FlushFile(sys.stdout)

    # '''To write to screen in real-time'''
    # message = lambda x: print(x, flush=True, end="")
    # message('I am flushing out now...')

    PORT = 8083
    HOST = 'localhost'

    delimiter = '--'
    read_params = False
    last_param = ''
    param_dict = {}

    print('sys.argv =', sys.argv)
    for arg in sys.argv:
        if arg == '--':
            read_params = True
        else:
            if read_params == True:
                if last_param == '':
                    last_param = arg
                else:
                    param_dict.update({last_param: arg})
                    last_param = ''

    print('param_dict =', param_dict)
    if 'PORT' in param_dict.keys():
        PORT =  int(param_dict['PORT'])
    if 'HOST' in param_dict.keys():
        HOST =  param_dict['HOST']


    print('Executing main with PORT=', PORT, '| HOST=', HOST)
    # sys.stdout.flush()
    main(PORT, HOST)