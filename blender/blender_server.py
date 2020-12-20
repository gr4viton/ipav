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


global param_dict
# HOST = "127.0.0.1"
# HOST = "192.168.1.100"
PATH_MAX = 4096

def printl():
    print('_'*42)

import bpy, mathutils, math
from mathutils import Vector
from math import pi

import time
from RealCamera import RealCamera

class BlenderServer():

    # real_cam_set
    real_cam_set = []
    # photogrammetry object
    pobj = None

    def __init__(self):
        # path = 'D:/DEV/PYTHON/pyCV/kivyCV_start/blender/real_cam_set.ini'
        # self.init_real_cam_set_from_conf(path)
        self.projections = None
        pass

    def init_real_cam_set(self):
        """ from blender model
        - loads position scale and rotation of all rcam.XXX
        """
        for obj_key, obj in bpy.data.objects.items():
            if obj_key[:4] == 'rcam':
                # print(obj_key)
                self.real_cam_set.append(RealCamera(
                    obj.name,
                    obj.location,
                    obj.rotation_euler,
                    'XYZ',
                    obj.dimensions,
                    obj
                ))
                # print(obj.location)

                # print(obj.rotation_euler)
                # print(obj.dimensions)

    def init_real_cam_set_from_conf(self, path):
        print ('Initializing real cameras location data.')
        vector_delimiter = '|'
        number_delimiter = ';'
        with open(path, 'r') as f:
            for cam_line in f.readlines():
                # print('cam_line =', cam_line)
                pos_rot_eul_size = []
                for vector in cam_line.split(vector_delimiter):
                    # print('vector =', vector)
                    name = None
                    if number_delimiter in vector:
                        pos_rot_eul_size.append( [float(value) for value in vector.split(number_delimiter)])
                    else:
                        if name == None:
                            name = vector
                        else:
                            pos_rot_eul_size.append(vector)
                    # print('pos_rot_size =', pos_rot_size[-1])

                self.real_cam_set.append(RealCamera(name, *pos_rot_eul_size))

        print(len(self.real_cam_set),'real camera location data initialized.')

    def print_objects(self):
        for obj in bpy.data.objects:
            print(obj)
        printl()

    def create_projections(self):
        if len(self.real_cam_set) < 1:
            return
        # self.real_cam_set[0].hull = self.projections

        #
        # hull_orig = [[[30, 31]],
        #              [[28, 33]],
        #              [[25, 34]],
        #              [[15, 34]],
        #              [[ 8, 31]],
        #              [[ 6, 30]],
        #              [[ 3, 27]],
        #              [[ 1, 24]],
        #              [[ 1, 20]],
        #              [[ 2, 18]],
        #              [[11,  9]],
        #              [[13,  8]],
        #              [[27,  8]],
        #              [[29, 10]],
        #              [[30, 16]]]

        # if self.projections:
        # #     print('\n'*5)
        # #     print(hull_orig)
        # #     print('%'*50)
        #     hull_orig = self.projections[0]
        #     # print(hull_orig)
        #     # print('\n'*5)
        # else:
        #     return
        #
        # hull = [x[0] for x in hull_orig]
        # print(hull)
        if self.projections:
            hull_orig = self.projections
        else:
            return


        pixel_len = 2
        hull = [hull_orig[i:i+pixel_len] for i in range(0, len(hull_orig), pixel_len)]
        # map (lambda x: hull_orig[pixel_len*x:(x+1)*pixel_len], range (pixel_len))
        # hull = hull_orig

        self.real_cam_set[0].hull = hull
        for rcam in self.real_cam_set:
            rcam.hull = hull

        for cam in self.real_cam_set:
            print('creating projection')
            cam.create_projection()

        return


    def duplicate(self, template, name):
        copy = template.copy()
        copy.name = name
        scene = bpy.context.scene
        scene.objects.link(copy)
        scene.update()
        return copy

    def photogrammetry_object(self):
        print('photogrammetrying object')
        bpy.ops.object.mode_set(mode='OBJECT')
        self.pobj = None

        # rcam.

        # print('real_cam_set', self.real_cam_set)
        # [print(rcam.proj) for rcam in self.real_cam_set]

        for rcam in self.real_cam_set:
            if rcam.proj:
                if self.pobj is None:
                    # first projection
                    print('first projection')
                    self.pobj = self.duplicate(rcam.proj, 'pobj')
                    # self.pobj = rcam.proj
                else:
                    print('boolean modifier')
                    # print(rcam.proj)
                    self.booleanSum(self.pobj, rcam.proj)
                    # print(rcam.proj)

        # [print(rcam.proj) for rcam in self.real_cam_set]

        # print('del first proj')
        # self.real_cam_set[0].delete_projection()

        # for rcam in self.real_cam_set:
        #     object = rcam.proj
        #
        #     scene = bpy.context.scene
        #     # [print(obj) for obj in scene.objects]
        #     # if object in scene.objects:
        #     scene.objects.unlink(object)
        #     bpy.data.objects.remove(object)
        #     scene.update()


        # [print(rcam.proj) for rcam in self.real_cam_set]
        for rcam in self.real_cam_set:
             rcam.delete_projection()

        scene = bpy.context.scene
        [print(obj) for obj in scene.objects]




    def execfile(self, filepath):
        import os
        global_namespace = {
            "__file__": filepath,
            "__name__": "__main__",
            }
        with open(filepath, 'rb') as file:
            exec(compile(file.read(), filepath, 'exec'), global_namespace)

    def booleanSum(self, target, other_object):

       bpy.ops.object.select_all(action='DESELECT')

       # Select the new other_object.
       target.select = True
       bpy.context.scene.objects.active = target

       other_object.select = True
       bpy.ops.object.make_single_user(object=True, obdata=True, material=False, texture=False, animation=False)
       # Add a modifier
       bpy.ops.object.modifier_add(type='BOOLEAN')

       mod = target.modifiers
       mod[0].name = "booleanSum"
       mod[0].object = other_object
       mod[0].operation = 'INTERSECT'

       # Apply modifier
       print('to apply modifier')
       bpy.ops.object.modifier_apply(apply_as='DATA', modifier=mod[0].name)


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


    def render_to_file(self):
        format_str = "%Y-%m-%d %H_%M_%S"
        # format_str = "-%H%M%S %Y-%m-%d"
        i = dt.datetime.now().strftime(format_str)
        path = os.path.abspath(''.join(['D:/DEV/PYTHON/pyCV/kivyCV_start/blender/pic/', str(i), ' image.jpg']))
        print('path =', path)
        bpy.data.scenes['Scene'].render.filepath = path
        bpy.ops.render.render( write_still=True )


    def save_to_mainfile(self):
        folder = os.path.abspath("D:\DEV\PYTHON\pyCV\_DELETE ME\pics")
        format_str = "%Y-%m-%d %H_%M_%S"
        i = dt.datetime.now().strftime(format_str)
        name = str(i) + 'out.blend'
        path = os.path.join(folder,name)
        bpy.ops.wm.save_as_mainfile(filepath=path)


    def load_file(self, blend):
        print("Loading file", blend)

        # multiple files opened simultaneously http://stackoverflow.com/questions/28075599/opening-blend-files-using-blenders-python-api
        path = "D:/DEV/PYTHON/pyCV/kivyCV_start/blender/main.blend"
        bpy.ops.wm.open_mainfile(filepath=path)
        sys.stdout.flush()

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
                    print('<<< loaded_pickle =', loaded_pickle)
                    try:
                        data_dict = pickle.loads(loaded_pickle)
                    except:
                        print('pickle loads failed')
                        continue
                    print('data_dict =', data_dict )
                    try:

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

                            if data_dict.get('init_real_cam_set', None) == True:
                                self.init_real_cam_set()

                            if data_dict.get('create_cam_projections', None) == True:
                                self.create_projections()

                            projections = data_dict.get('projections', None)
                            if projections:
                                self.projections = projections

                            if data_dict.get('photogrammetry_object', None) == True:
                                self.photogrammetry_object()

                            if data_dict.get('render', None) == True:
                                self.render_to_file()

                            if data_dict.get('save_blend', None) == True:
                                self.save_to_mainfile()



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

                    except OSError as err:
                        print("OS error: {0}".format(err))
                    # except :
                    #     print("Unexpected error:", sys.exc_info()[0])
                        # print(sys.exc_traceback)
                        # print(sys.exc_value)


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
