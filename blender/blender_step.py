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
# import select

import pickle

#TODO
# socket communication with script running in blender (scipt running in blender would be the blender_server)
# multiple blender servers - at the same time (synchro would be difficult)
# pickle encapsuling data - with custom protocol version (python 3.4.2 in blender 2.7.6)
# do not send the whole pickle every time - the other side should buffer it, so only changes should be sent
# keyword marshalling

class blender_module():

    PORT = 8084
    HOST = "localhost"
    param_dict = {}
    param_exit = 'exit()'

    def __init__(self):
        self.init_path()
        self.init_params()

        self.data_pickle = pickle.dumps(self.param_dict)
        print('data_pickle =', self.data_pickle)


    def init_path(self):
        blender_dir = os.path.abspath('C:/PROG/grafic/Blender/')
        blender_file = 'blender.exe'
        blender_path = os.path.join(blender_dir, blender_file)

        script_dir = os.path.abspath('D:/DEV/PYTHON/pyCV/kivyCV_start/blender')
        self.pic_dir = os.path.join(script_dir, 'pic')


        self.script_path = os.path.join(script_dir, 'render.py')
        self.blend_path = os.path.join(script_dir, 'main.blend')


        # script_params = ' '.join([script_path, pic_dir])

        blender_server_path = os.path.join(script_dir, 'blender_server.py')

        other_params = ' '.join(['-noglsl', '-noaudio', '-nojoystick'])
        # self.blender_server = ' '.join([blender_path, '-b', other_params,'--python', blender_server_path])

        self.blender_server = ' '.join([blender_path, '-b', '--python', blender_server_path])
        # self.blender_server = ' '.join([blender_path, '--python', blender_server_path])
        print('blender_server = ', self.blender_server)

    def init_params(self):

        # param dict
        self.param_dict.update({'PORT': self.PORT})
        self.param_dict.update({'HOST': self.HOST})

        str_hostport = ' '.join([' '.join([str(key), str(value)])
                                   for key, value in self.param_dict.items()])

        self.param_dict.update({'pic_dir': self.pic_dir})

        print('str_param_dict =', str_hostport)


        self.blender_server_params = ' '.join([self.blender_server, '--', str_hostport])
        print('blender_server_params =', self.blender_server_params)

    def send_string_command(self, param):
        clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        clientsocket.connect((self.HOST, self.PORT))

        # for arg in sys.argv[1:]:
        # for arg in argv:
        clientsocket.sendall(param.encode("utf-8") + b'\x00')

    def send_pickle(self, pickle):
        clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        clientsocket.connect((self.HOST, self.PORT))
        print('>>> sending pickle=', pickle)
        clientsocket.sendall(pickle + b'\x00' )

    def send_data_dict(self, data_dict):
        print('>>> pickling data_dict=', data_dict)
        self.send_pickle(pickle.dumps(data_dict))

    def delete_image_files(self):
        path = self.pic_dir
        for the_file in os.listdir(path):
            file_path = os.path.join(path, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                #elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)

    def start_server(self):
        print('%'*42, 'blendering')
        # 'blender myscene.blend --background --python myscript.py'


        # self.blender = Popen(blender_path, stdout=PIPE, stdin=PIPE, stderr=STDOUT) #, shell=False)
        self.bs = Popen(self.blender_server_params, stdin=PIPE, stderr=STDOUT) #, shell=False)
        # must be opened as binary - to ensure correct pickle transfer

        print('blender_server process started!')
        # blender.stdin.write(bytes(params, 'UTF-8'))

    def init_room(self):
        self.send_data_dict({'load_blend': self.blend_path})
        time.sleep(1)
        self.send_data_dict({'init_real_cam_set':True})

    def photogrammetry_object(self, projections):
        start = time.time()

        self.send_data_dict({'create_cam_projections': True})
        self.send_data_dict({'photogrammetry_object': True})
        self.send_data_dict({'render': True})

        # time.sleep(1)
        end = time.time()
        print('pickling data took enormous time [s]:', end-start)

        im = os.path.abspath('D:\\DEV\\PYTHON\\pyCV\\kivyCV_start\\blender\\pic\\')
        return im



    def run_commands(self):

        self.send_data_dict({'load_blend': self.blend_path})
        self.send_data_dict({'init_real_cam_set':True})
        # time.sleep(10)
        self.send_data_dict({'create_cam_projections':True})
        self.send_data_dict({'photogrammetry_object':True})


        looping = True
        # a = 5
        a = 1
        while(looping):

            self.send_data_dict({'render':True})

            time.sleep(0.05)
            time.sleep(1)
            # time.sleep(4)

            if a > 1:
                a -= 1
            else:
                self.send_data_dict({'exit':True})
                looping = False

        time.sleep(3)
        self.delete_image_files()

    def run(self):
        self.start_server()
        time.sleep(2)
        # should wait from started tag from server
        self.run_commands()


    # import_scene

if __name__ == '__main__':
    bm = blender_module()
    bm.run()