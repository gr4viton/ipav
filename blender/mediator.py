
import bpy, mathutils, math
from mathutils import Vector
from math import pi

import os
import sys
import fileinput
import time

import stat
import select



# script_dir = os.path.abspath('D:/DEV/PYTHON/pyCV/blender_out/')
script_dir = os.path.abspath('D:/DEV/PYTHON/pyCV/kivyCV_start/blender')
script_file = 'render.py'
# script_file = 'mediator.py'
script_path = os.path.join(script_dir, script_file)

filename = script_path
script= compile(open(filename).read(), filename, 'exec')

# def start_script(script):
#     hey()
#     print('Starting script:')
#     print(script)
#
#     exec(script) in globals(), locals()
#     # try:
#     #     exec(compile(open(filename).read(), filename, 'exec'))
#     # except(ex):
#     #     print(ex)
#
# def render(origin):
#     start_script(script_path)

def hey():
    print("MEDIATOR HEY!")

def printl():
    print('_'*42)


if __name__ == "__main__":
    printl()
    print('Waiting for commands from stdin.')
    printl()

    # exec(script)
    looping = True
    while looping:
        # print('looping')
        # select((sys.stdin,),(),())

        # print(sys.stdin)
        # sys.stdout.flush()

        # line = sys.stdin.readline()
        # for line in sys.stdin:
        #     print(line)

        # print('input')
        if not sys.stdin.isatty():
            print("Empty stdin")
        if sys.stdin.isatty():
            # print("is  sys.stdin.isatty")
            print('meh')
            for line in sys.stdin:
                print(line)
                if line == 'exit':
                    looping = False
                elif line == 'render':
                    exec(script)


        mode = os.fstat(sys.stdin.fileno()).st_mode
        if stat.S_ISFIFO(mode):
             print("stdin is piped")
        elif stat.S_ISREG(mode):
             print( "stdin is redirected")
        else:
             print ("stdin is terminal")



        while True:
            input = ""
            print('toread')
            sys.stdout.flush()
            c = sys.stdin.read(1)
            print('have read')
            sys.stdout.flush()
            while c is not sys.EOF:
                input += c
                c = sys.stdin.read(1)
            for line in input.split('\n'):
                print(line)

        # k = 0
        # try:
        #     buff = ''
        #     while True:
        #         buff += sys.stdin.read(1)
        #         if buff.endswith('\n'):
        #             print(buff[:-1])
        #             buff = ''
        #             k = k + 1
        # except KeyboardInterrupt:
        #    sys.stdout.flush()
        #    pass
        # print(k)










        time.sleep(1)
        sys.stdout.flush()

    # start_script(script_compiled)
    # render.run(Vector((0,0,0)))
    printl()
    print('End of com.py')
    printl()
