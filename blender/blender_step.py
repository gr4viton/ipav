import numpy as np
import cv2

import findHomeography as fh
import sys
import time
import os
from subprocess import Popen, PIPE, STDOUT

# import blender_client
import sys
import socket
import select

class blender_module():

    blender_dif = os.path.abspath('C:/PROG/grafic/Blender/')
    blender_file = 'blender.exe'
    blender_path = os.path.join(blender_dif, blender_file)


    # script_dir = os.path.abspath('D:/DEV/PYTHON/pyCV/blender_out/')
    script_dir = os.path.abspath('D:/DEV/PYTHON/pyCV/kivyCV_start/blender')
    script_file = 'render.py'
    # script_file = 'mediator.py'
    script_path = os.path.join(script_dir, script_file)

    blender_server_path = os.path.join(script_dir, 'blender_server.py')
    blender_server = blender_path + ' -b --python ' + blender_server_path


    PORT = 8081
    HOST = "localhost"

    def send_command(self, argv):
        clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        clientsocket.connect((self.HOST, self.PORT))

        # for arg in sys.argv[1:]:
        for arg in argv:
            clientsocket.sendall(arg.encode("utf-8") + b'\x00')


    def run(self):
        print('%'*42, 'blendering')
        # 'blender myscene.blend --background --python myscript.py'


        # self.blender = Popen(blender_path, stdout=PIPE, stdin=PIPE, stderr=STDOUT) #, shell=False)
        self.bs = Popen(self.blender_server, stdout=PIPE, stdin=PIPE, stderr=STDOUT) #, shell=False)

        print('blender_server running and waiting for command!')
        # blender.stdin.write(bytes(params, 'UTF-8'))

        # self.print_stdout()

        time.sleep(4)

        looping = True
        while(looping):
            print('sending command >> ', self.script_path)
            self.send_command(self.script_path)
            #
            # param_bytes = params.encode('utf-8')
            # # grep_stdout = self.blender.communicate(input=param_bytes)[0]
            # self.blender.stdin.write(param_bytes)
            # grep_stdout = self.blender.communicate()[0]
            # print(param_bytes)
            # print(grep_stdout)
            #
            # sys.stdout.flush()
            time.sleep(3.5)

        # self.blender.stdin.close()


    # import_scene

if __name__ == '__main__':
    bm = blender_module()
    bm.run()