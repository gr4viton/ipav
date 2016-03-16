import numpy as np
import cv2

import findHomeography as fh
import sys
import time
import os
from subprocess import Popen, PIPE


class blender_module():

    def print_stdout(self):
        printing = True
        while printing:
            out = self.blender.stdout.readline()
            print(out[:-1].decode('utf-8'))
            if not self.blender.stdout.isatty():
                printing = False

    def write_stdin(self, params):
        # self.blender.stdin.write(bytes(params, 'UTF-8'))
        # params += '\n\r'
        self.blender.stdin.write(params.encode('utf-8'))
        self.blender.stdin.flush()



    def run(self):
        print('%'*42, 'blendering')
        # 'blender myscene.blend --background --python myscript.py'
        blender_dif = os.path.abspath('C:/PROG/grafic/Blender/')
        blender_file = 'blender.exe'
        blender_path = os.path.join(blender_dif, blender_file)

        script_dir = os.path.abspath('D:/DEV/PYTHON/pyCV/blender_out/')
        script_file = 'render.py'
        script_file = 'mediator.py'
        script_path = os.path.join(script_dir, script_file)

        params = str(' '.join([' -b', '--python', '"' + script_path + '"']))
        blender_path += params
        print(params)
        print(blender_path)
        self.blender = Popen(blender_path, stdout=PIPE, stdin=PIPE, stderr=PIPE) #, shell=False)

        print('a')

        # blender.stdin.write(bytes(params, 'UTF-8'))

        self.print_stdout()

        time.sleep(2)

        looping = True
        while(looping):

            params = 'exit'

            self.write_stdin(params)
            # print('sent data :', params)

            # params = 'render'
            # write_stdin(params)

            self.print_stdout()

            sys.stdout.flush()
            time.sleep(0.05)





    # import_scene

if __name__ == '__main__':
    bm = blender_module()
    bm.run()