import numpy as np
import cv2

import findHomeography as fh
import sys
import time
import os
# delting files
import shutil
from subprocess import Popen, PIPE, STDOUT

# import blender_client
import sys
import socket
import select

import pickle

#TODO
# socket communication with script running in blender (scipt running in blender would be the blender_server)
# multiple blender servers - at the same time (synchro would be difficult)
# pickle - with custom protocol version (blender python version 3.4 ?) encapsuling data

class blender_module():

    PORT = 8084
    HOST = "localhost"
    param_dict = {}
    param_exit = 'exit()'

    def __init__(self):
        self.init_path()
        self.init_params()

        self.data_pickle = pickle.dumps(self.param_dict)


    def init_path(self):
        blender_dir = os.path.abspath('C:/PROG/grafic/Blender/')
        blender_file = 'blender.exe'
        blender_path = os.path.join(blender_dir, blender_file)

        script_dir = os.path.abspath('D:/DEV/PYTHON/pyCV/kivyCV_start/blender')
        self.pic_dir = os.path.join(script_dir, 'pic')
        script_file = 'render.py'
        # script_file = 'mediator.py'
        self.script_path = os.path.join(script_dir, script_file)

        # script_params = ' '.join([script_path, pic_dir])

        blender_server_path = os.path.join(script_dir, 'blender_server.py')

        other_params = ' '.join(['-noglsl', '-noaudio', '-nojoystick'])
        # self.blender_server = ' '.join([blender_path, '-b', other_params,'--python', blender_server_path])

        self.blender_server = ' '.join([blender_path, '-b', '--python', blender_server_path])
        print('blender_server = ', self.blender_server)

    def init_params(self):

        # param dict
        self.param_dict.update({'PORT': self.PORT})
        self.param_dict.update({'HOST': self.HOST})
        self.param_dict.update({'pic_dir': self.pic_dir})

        str_param_dict = ' '.join([' '.join([str(key), str(value)])
                                   for key, value in self.param_dict.items()])
        print('str_param_dict =', str_param_dict)


        self.blender_server_params = ' '.join([self.blender_server, '--', str_param_dict])
        print('blender_server_params =', self.blender_server_params)

    def send_command(self, param):
        clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        clientsocket.connect((self.HOST, self.PORT))

        # for arg in sys.argv[1:]:
        # for arg in argv:
        clientsocket.sendall(param.encode("utf-8") + b'\x00')


    def run(self):
        print('%'*42, 'blendering')
        # 'blender myscene.blend --background --python myscript.py'


        # self.blender = Popen(blender_path, stdout=PIPE, stdin=PIPE, stderr=STDOUT) #, shell=False)
        self.bs = Popen(self.blender_server_params, stdin=PIPE, stderr=STDOUT) #, shell=False)
        # must be opened as binary - to ensure correct pickle transfer

        print('blender_server process started!')
        # blender.stdin.write(bytes(params, 'UTF-8'))

        # self.print_stdout()

        time.sleep(2)
        # should wait from started tag from server

        looping = True
        a = 5
        while(looping):

            # param = self.script_params
            param = self.script_path
            print('sending command >> ', param)
            # param_bytes = param.encode('utf-8')

            self.send_command(param)


            # self.bs.communicate()
            # print(txt)
            #
            # param_bytes = params.encode('utf-8')
            # # grep_stdout = self.blender.communicate(input=param_bytes)[0]
            # self.blender.stdin.write(param_bytes)
            # grep_stdout = self.blender.communicate()[0]
            # print(param_bytes)
            # print(grep_stdout)
            #
            # sys.stdout.flush()

            time.sleep(0.05)
            # time.sleep(4)

            if a > 0:
                a -= 1
            else:
                self.send_command(self.param_exit)
                looping = False


        # self.blender.stdin.close()
        time.sleep(1)
        self.delete_images()

    def delete_images(self):
        path = self.pic_dir
        for the_file in os.listdir(path):
            file_path = os.path.join(path, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                #elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)

    # import_scene

if __name__ == '__main__':
    bm = blender_module()
    bm.run()