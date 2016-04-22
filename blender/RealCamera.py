
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

    proj = None
    rcam = None

    def __init__(self, pos, rot, size):
        self.pos = Vector(pos)
        self.rot = Vector(rot)
        self.size = Vector(size)

    def __init__(self, pos, rot, size, rcam_obj):
        self.pos = Vector(pos)
        self.rot = Vector(rot)
        self.size = Vector(size)
        self.rcam = rcam_obj

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

    def create_cone(self):
        new_cone = bpy.ops.mesh.primitive_cone_add(
            vertices=4, radius1=2, depth=10.0,
            location=self.pos, rotation=self.rot)
        print("created new_cone =", new_cone)
        # scn = bpy.context.scene.GetCurrent()
        # scn.objects.new(new_cone, 'cone')

    def delete_object(self, object):
    #     bpy.ops.object.select_all(action='DESELECT')
    #     #rcam_d = bpy.data.objects['prj.001']
    #     object.select = True
    #     bpy.ops.object.delete()
    #
    #     scene = bpy.context.scene
    # #    scene.objects.link(rcam_d)
    #     scene.update()


        scene = bpy.context.scene
        scene.objects.unlink(object)
        bpy.data.objects.remove(object)
        scene.update()

        # object.hide
        # when it is unlinked from scene, it won't be saved
        object = None

        # object.delete()

    def delete_projection(self):
        self.delete_object(self.proj)
        self.proj = None

    def create_projection(self, hull=None):
        if self.rcam:
            if self.proj:
                self.delete_projection()

            if hull == None:
                self.proj = self.rcam.copy()
                self.proj.scale = Vector((1, 1, 5))

            else:
                verts = [(1.0, 1.0, -1.0),
                        (1.0, -1.0, -1.0),
                        (-1.0, -1.0, -1.0),
                        (-1.0, 1.0, -1.0),
                         (1.0, 1.0, 1.0),
                         (1.0, -1.0, 1.0),
                        (-1.0, -1.0, 1.0),
                        (-1.0, 1.0, 1.0)]

                faces = [(0, 1, 2, 3),
                         (4, 7, 6, 5),
                         (0, 4, 5, 1),
                         (1, 5, 6, 2),
                         (2, 6, 7, 3),
                         (4, 0, 3, 7)]

                mesh_data = bpy.data.meshes.new("cube_mesh_data")
                mesh_data.from_pydata(verts, [], faces)
                mesh_data.update()

                self.proj = bpy.data.objects.new("My_Object", mesh_data)
                pass


            self.proj.name = 'prj' + self.rcam.name[-4:]
            #    rcam_d.location += Vector((1,1,1))
            #    rcam_d.rotation_euler = Vector((pi,0,0))

            scene = bpy.context.scene
            scene.objects.link(self.proj)
            scene.update()



# import bpy, bmesh
#
# verts = ((0, 3),(2.5, 0.5),(5, 1),(4.5, 3.5),(10.5, 2),(8, 10),(7, 4.5),(2, 6))
#
# bm = bmesh.new()
# for v in verts:
#     bm.verts.new((v[0], v[1], 0))
# bm.faces.new(bm.verts)
#
# bm.normal_update()
#
# me = bpy.data.meshes.new("")
# bm.to_mesh(me)
#
# ob = bpy.data.objects.new("", me)
# bpy.context.scene.objects.link(ob)
# bpy.context.scene.update()