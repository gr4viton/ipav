# http://blender.stackexchange.com/questions/41533/how-to-remotely-run-a-python-script-in-an-existing-blender-instance

# Script to run from blender:
#   blender --python blender_server.py

PORT = 8083
HOST = "localhost"
# HOST = "127.0.0.1"
# HOST = "192.168.1.100"
PATH_MAX = 4096

import sys

def execfile(filepath):
    import os
    global_namespace = {
        "__file__": filepath,
        "__name__": "__main__",
        }
    with open(filepath, 'rb') as file:
        exec(compile(file.read(), filepath, 'exec'), global_namespace)


def main():
    import socket

    serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serversocket.bind((HOST, PORT))
    serversocket.listen(1)

    print("Listening on %s:%s" % (HOST, PORT))
    while True:
        connection, address = serversocket.accept()
        buf = connection.recv(PATH_MAX)

        for filepath in buf.split(b'\x00'):
            if filepath:
                print("Executing:", filepath)
                # sys.stderr.write(str(''.join(["Executing:", filepath])))
                try:
                    execfile(filepath)
                except:
                    import traceback
                    traceback.print_exc()


if __name__ == "__main__":
    main()