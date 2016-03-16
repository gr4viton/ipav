import numpy as np
import cv2

import findHomeography as fh
import sys
import time
import os
from subprocess import Popen, PIPE, STDOUT

import select

class blender_module():
    outs = ['-','-','-']

    def print_stdout(self):
        printing = True

        while printing:
            out = self.blender.stdout.readline()
            str_out = out[:-1].decode('utf-8')
            self.outs.append(str_out)

            # if not self.blender.stdout.isatty():
            #     return

            # print(self.outs[-1])
            a = ''.join([self.outs[-1], self.outs[-2], self.outs[-3]])
            # print("x"*5, a, "y"*5)
            if a == '':
                return

            print(str_out)

    def write_stdin(self, params):
        # self.blender.stdin.write(bytes(params, 'UTF-8'))
        # params += '\n\r'
        # self.blender.stdin.write(params.encode('utf-8'))

        param_bytes = params.encode('utf-8')
        self.blender.stdin.write(param_bytes)

        # self.blender.communicate(input=param_bytes)

        # self.blender.stdin.flush()
        print('>>> writed >>> ',params)



    def run(self):
        print('%'*42, 'blendering')
        # 'blender myscene.blend --background --python myscript.py'
        blender_dif = os.path.abspath('C:/PROG/grafic/Blender/')
        blender_file = 'blender.exe'
        blender_path = os.path.join(blender_dif, blender_file)

        # script_dir = os.path.abspath('D:/DEV/PYTHON/pyCV/blender_out/')
        script_dir = os.path.abspath('D:/DEV/PYTHON/pyCV/kivyCV_start/blender')
        script_file = 'render.py'
        script_file = 'mediator.py'
        script_path = os.path.join(script_dir, script_file)

        params = str(' '.join([' -b', '--python', '"' + script_path + '"']))
        blender_path += params
        print(params)
        print(blender_path)
        self.blender = Popen(blender_path, stdout=PIPE, stdin=PIPE, stderr=STDOUT) #, shell=False)

        # blender.stdin.write(bytes(params, 'UTF-8'))

        self.print_stdout()

        time.sleep(2)

        looping = True
        while(looping):

            params = 'exit'
            # params = 'render'

            # self.write_stdin(params)
            # self.print_stdout()

            param_bytes = params.encode('utf-8')
            # grep_stdout = self.blender.communicate(input=param_bytes)[0]
            self.blender.stdin.write(param_bytes)
            grep_stdout = self.blender.communicate()[0]
            print(grep_stdout)

            sys.stdout.flush()
            time.sleep(0.5)

        self.blender.stdin.close()



    # import_scene

if __name__ == '__main__':
    bm = blender_module()
    bm.run()