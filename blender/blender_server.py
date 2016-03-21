# http://blender.stackexchange.com/questions/41533/how-to-remotely-run-a-python-script-in-an-existing-blender-instance

# Script to run from blender:
#   blender --python blender_server.py

import sys
import pickle

import bpy
import os
import datetime as dt



sys.path.append("D:/DEV/PYTHON/pyCV/kivyCV_start/blender")
import render


# HOST = "127.0.0.1"
# HOST = "192.168.1.100"
PATH_MAX = 4096

def printl():
    print('_'*42)

import bpy, mathutils, math
from mathutils import Vector
from math import pi

import time


class RealCamera():
    pos = Vector((0,0,0))
    rot = Vector((0,0,0))
    # height, width, distance of the plane == aspect ratios, distance
    size = Vector((0,0,0))

    def __init__(self, pos, rot, size):
        self.pos = Vector(pos)
        self.rot = Vector(rot)
        self.size = Vector(size)

    def init_rot(self, real_point, pixel):
        """
        initializes rotation from real space point projection to camera plane
        """
    def create_iso(self):
        cubeobject = bpy.ops.mesh.primitive_ico_sphere_add
        #    cursor = context.scene.cursor_location
        #   x = cursor.x
        #  y = cursor.y
        # z = cursor.z
        cubeobject(location=(1, 1, 1))

    def create_line(self):
        new_cone = bpy.ops.mesh.primitive_cone_add(
            vertices=32, radius1=2, depth=10.0,
            location=self.pos, rotation=self.rot)


        print("created new_cone =", new_cone)
        # scn = bpy.context.scene.GetCurrent()
        # scn.objects.new(new_cone, 'cone')




class BlenderServer():

    # real_cam_set
    real_cam_set = []

    def __init__(self):
        path = 'D:/DEV/PYTHON/pyCV/kivyCV_start/blender/real_cam_set.ini'
        self.init_real_cam_set(path)


    def init_real_cam_set(self, path):
        print ('Initializing real cameras location data.')
        vector_delimiter = '|'
        number_delimiter = ';'
        with open(path, 'r') as f:
            for cam_line in f.readlines():
                # print('cam_line =', cam_line)
                pos_rot_size = []
                for vector in cam_line.split(vector_delimiter):
                    # print('vector =', vector)
                    pos_rot_size.append( [float(value) for value in vector.split(number_delimiter)])
                    # print('pos_rot_size =', pos_rot_size[-1])

                self.real_cam_set.append(RealCamera(*pos_rot_size))

        print(len(self.real_cam_set),'real camera location data initialized.')

    def create_cam_lines(self):
        for cam in self.real_cam_set:
            print('creating line')
            cam.create_line()

    def execfile(self, filepath):
        import os
        global_namespace = {
            "__file__": filepath,
            "__name__": "__main__",
            }
        with open(filepath, 'rb') as file:
            exec(compile(file.read(), filepath, 'exec'), global_namespace)

    def csgSubtract(self, target, opObj):
       '''subtract opObj from the target'''

       # Deselect All
       bpy.ops.object.select_all(action='DESELECT')

       # Select the new object.
       target.select = True
       bpy.context.scene.objects.active = target

       # Add a modifier
       bpy.ops.object.modifier_add(type='BOOLEAN')

       mod = target.modifiers
       mod[0].name = "SubEmUp"
       mod[0].object = opObj
       mod[0].operation = 'DIFFERENCE'

       # Apply modifier
       bpy.ops.object.modifier_apply(apply_as='DATA', modifier=mod[0].name)


    def bisect(self):
        """
        if it is quicker than subtract - bisect individual planes
        """


        pass



    def load_file(self, blend):
        print("Loading file", blend)

        # multiple files opened simultaneously http://stackoverflow.com/questions/28075599/opening-blend-files-using-blenders-python-api
        path = "D:/DEV/PYTHON/pyCV/kivyCV_start/blender/main.blend"
        bpy.ops.wm.open_mainfile(filepath=path)
        sys.stdout.flush()

    def render_to_file(self):
        format_str = "%Y-%m-%d %H_%M_%S"
        # format_str = "-%H%M%S %Y-%m-%d"
        i = dt.datetime.now().strftime(format_str)
        path = os.path.abspath(''.join(['D:/DEV/PYTHON/pyCV/kivyCV_start/blender/pic/', str(i), ' image.jpg']))
        print('path =', path)
        bpy.data.scenes['Scene'].render.filepath = path
        bpy.ops.render.render( write_still=True )

    def init_render(self):
        self.width = 100
        self.height = 100
        bpy.data.scenes['Scene'].render.filepath='file1' # directory and name
        bpy.data.scenes['Scene'].render.resolution_x = self.width
        bpy.data.scenes['Scene'].render.resolution_y = self.height
        bpy.data.scenes['Scene'].render.pixel_aspect_x = 1.0
        bpy.data.scenes['Scene'].render.pixel_aspect_y = 1.0

    def render(self):
        image = bpy.data.images['image02'] # image02 as seen in uv editor
        imageR = bpy.data.images['Render Result'] # useless, so bad
        width = image.size[0]
        height = image.size[1]

        PIXELS = [0.0 for i in range(len(image.pixels))]
        # len(image.pixels) == width * height * 3 ( or 4 with the alpha channel )

        # here, work with PIXELS
        image.pixels=PIXELS

        # nodes work around - only with blender gui on - no other work arounds when gui is off
        # https://ammous88.wordpress.com/2015/01/16/blender-access-render-results-pixels-directly-from-python-2/

    def main(self, PORT, HOST):
        import socket

        serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        serversocket.bind((HOST, PORT))
        serversocket.listen(1)

        print("Listening on %s:%s" % (HOST, PORT))
        printl()

        looping = True
        ending = pickle.STOP + b'\x00'
        while looping:
            connection, address = serversocket.accept()
            buf = connection.recv(PATH_MAX)

            # for loaded_pickle in buf.split(b'\x00'):
            for loaded_pickle in buf.split(ending):

                if loaded_pickle:
                    loaded_pickle += pickle.STOP
                    print('loaded_pickle =', loaded_pickle)
                    data_dict = pickle.loads(loaded_pickle)
                    print('data_dict =', data_dict )

                    if data_dict:
                        script = data_dict.get('exec', None)
                        if script:
                            print("Executing:", script)
                            self.execfile(script)

                        blend = data_dict.get('load_blend', None)
                        if blend:
                            self.load_file(blend)


                        # exitit = data_dict.get('exit', None)
                        # if exitit == True:
                        #     looping = False
                        #     break

                        if data_dict.get('create_cam_lines', None) == True:
                            self.create_cam_lines()

                        if data_dict.get('render', None) == True:
                            self.render_to_file()

                        if data_dict.get('exit', None) == True:
                            looping = False
                            break

                        # if 'exit' in data_dict:
                        #     if data_dict['exit'] == True:
                        #         looping = False
                        #         break

                        # elif 'exec' in data_dict:
                        #     script = data_dict.get('exec')
                        #     print("Executing:", script)
                        #     execfile(script)
                        else:
                            printl()

                # sys.stdout.flush()

            # for filepath in buf.split(b'\x00'):
            #     if filepath:
            #         if filepath == b'exit()':
            #             print('blender_server got exit() command thus shutting down blender program.')
            #             looping = False
            #             break
            #         else:
            #             print("Executing:", filepath)
            #             # sys.stderr.write(str(''.join(["Executing:", filepath])))
            #             try:
            #                 execfile(filepath)
            #             except:
            #                 import traceback
            #                 traceback.print_exc()

        serversocket.close()

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
    bs = BlenderServer()
    bs.main(PORT, HOST)

